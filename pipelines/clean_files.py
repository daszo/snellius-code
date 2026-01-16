import sqlite3
import pandas as pd
import re
import numpy as np
from CE.utils.database import finish_clean_message_and_drop_folders

# --- CONSTANTS ---
# Defined globally so both splitting and truncation use the exact same logic
HISTORY_DELIMITERS = [
    r"-----Original Message-----",
    r"----- Forwarded by",
    r"^\s*From:\s.+?Sent:\s.+?To:\s",  # Outlook Header quote
    r"^_+?$",  # Underscore line
    r'^".*?"\s+<.*?>\s+on\s+\d{2}/\d{2}/\d{4}',  # Lotus Notes style
]


def remove_legal_disclaimer(text: str) -> str:
    """(USER LOGIC) Removes specific legal disclaimer."""
    disclaimer_text = """This e-mail message may contain legally privileged and/or confidential
    information. If you are not the intended recipient(s), or the employee
    or agent responsible for delivery of this message to the intended
    recipient(s), you are hereby notified that any dissemination,
    distribution or copying of this e-mail message is strictly prohibited.
    If you have received this message in error, please immediately notify
    the sender and delete this e-mail message from your computer."""

    pattern_parts = [re.escape(word) for word in disclaimer_text.split()]
    regex_pattern = r"\s*".join(pattern_parts)
    compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)

    return compiled_pattern.sub("", text).strip()


def prune_signature_footer(text: str) -> str:
    """(GEMINI LOGIC) Reverse scan to eat the signature block."""
    if not text:
        return ""

    lines = text.splitlines()

    clutter_patterns = [
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",
        r"\b(fax|cell|mobile|office|w|c|h)\b",
        r"\b(Street|St|Ave|Rd|Blvd|Suite|Floor|Haymarket)\b",
        r"\b\d{5}\b",
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
        r"\b(London|Houston|New York|Calgary|Portland)\b",
        r"@enron\.com",
        r"Best Regards,",
        r"Sincerely,",
        r"Thanks,",
        r"Regards,",
    ]

    content_keepers = [r"[.?!]$", r"^(?!\s*$).{60,}$"]

    final_lines = []
    is_cleaning_footer = True

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if not is_cleaning_footer:
                final_lines.append(line)
            continue

        is_clutter = any(
            re.search(p, stripped, re.IGNORECASE) for p in clutter_patterns
        )
        is_keeper = any(re.search(p, stripped) for p in content_keepers)

        if is_cleaning_footer:
            if is_clutter and not is_keeper:
                continue
            else:
                is_cleaning_footer = False
                final_lines.append(line)
        else:
            final_lines.append(line)

    return "\n".join(reversed(final_lines)).strip()


def clean_segment_content(text: str) -> str:
    """
    (REFACTORED) Applies the regex scrubbing and signature pruning.
    This logic is now separate from the history truncation so it can be
    applied to history segments too.
    """
    if not text:
        return ""

    # --- Regex Scrubbing (USER LOGIC) ---
    text = remove_legal_disclaimer(text)
    text = re.sub(r"\*{10,}[\s\S]*?\*{10,}", "", text)  # Enron block
    text = re.sub(r"^[\s>]+", "", text, flags=re.MULTILINE)  # Quotes

    # Header Blocks
    header_block_pattern = r"(?:^.+? on \d{2}/\d{2}/\d{4}[\s\S]*?|^\s*(?:From|Sent|To|Cc|Please respond to):[\s\S]*?)^Subject:.*$"
    text = re.sub(header_block_pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Labeled Contact Info
    text = re.sub(
        r"\b(?:Fax|Direct|Tel)\s*:\s*[+\d\s().-]+", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\bEmail\s*:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Generic Info
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    text = re.sub(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext)\.?\s*\d+)?",
        "",
        text,
    )

    # Symbols
    text = re.sub(r"(?:\s|^)-+(?=\s|$)|-{2,}", "", text)
    text = re.sub(r"\b\w+(?:\.[a-zA-Z]\w*)+\b", "", text)
    text = re.sub(r"^\s*\?+\s*(?:\r?\n|\r)?", "", text, flags=re.MULTILINE)

    # --- Signature Pruning (GEMINI LOGIC) ---
    text = prune_signature_footer(text)

    # Final Polish
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ==========================================
#   SPLITTING & TRUNCATION LOGIC
# ==========================================


def truncate_email_history(text: str) -> str:
    """(ORIGINAL) Returns ONLY the newest message, cutting at the first delimiter."""
    if not text:
        return ""
    lines = text.splitlines()
    cut_lines = []
    for line in lines:
        if any(re.search(p, line, re.IGNORECASE) for p in HISTORY_DELIMITERS):
            break
        cut_lines.append(line)
    return "\n".join(cut_lines)


def split_email_chain(text: str) -> list[str]:
    """
    (NEW) Splits an email into a list of messages [Newest, Reply1, Reply2...]
    based on the history delimiters.
    """
    if not text:
        return []

    lines = text.splitlines()
    segments = []
    current_segment = []

    for line in lines:
        # Check if this line marks the start of a previous message
        if any(re.search(p, line, re.IGNORECASE) for p in HISTORY_DELIMITERS):
            # If we have content in the current buffer, save it
            if current_segment:
                segments.append("\n".join(current_segment))
            # Start a new segment with this line (it's the start of the next header)
            current_segment = [line]
        else:
            current_segment.append(line)

    # Append the final segment
    if current_segment:
        segments.append("\n".join(current_segment))

    return segments


# ==========================================
#   MAIN FUNCTIONS
# ==========================================


def clean_email_body(text: str) -> str:
    """
    OPTION A: The Original Behavior
    Returns cleaned text of ONLY the newest message.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Truncate history (throw away old messages)
    text = truncate_email_history(text)

    # 2. Clean the remainder
    return clean_segment_content(text)


def clean_full_chain(text: str) -> str:
    """
    OPTION B: The New Requirement
    Splits the chain, cleans EACH message individually, and concatenates them.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Split into segments [Newest, Old_1, Old_2...]
    raw_segments = split_email_chain(text)

    # 2. Clean each segment individually
    cleaned_segments = [clean_segment_content(seg) for seg in raw_segments]

    # 3. Filter out empty segments (in case a segment was just signatures)
    cleaned_segments = [seg for seg in cleaned_segments if seg]

    # 4. Join with double newline
    return "\n\n".join(cleaned_segments)


# ==========================================
#   PIPELINE
# ==========================================


def clean_email_bodies_pipeline(DB_PATH="data/enron.db", keep_full_history=False):
    """
    Added 'keep_full_history' flag to toggle between behavior A and B.
    """
    print("starting to load")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM Message", conn)
    conn.close()
    print(f"loaded database {DB_PATH}")

    # --- SELECT CLEANING STRATEGY ---
    if keep_full_history:
        print("Applying FULL CHAIN cleaning (Newest + History)...")
        cleaning_func = clean_full_chain
    else:
        print("Applying TRUNCATED cleaning (Newest Only)...")
        cleaning_func = clean_email_body

    df["body_clean"] = df["body"].apply(cleaning_func)
    df = df[df["body_clean"].str.strip() != ""]

    print(f"Cleaned database {DB_PATH} of shape {df.shape}")

    # (Remaining pipeline logic unchanged...)
    df["clean_length_character"] = df["body_clean"].str.len()
    df["clean_length_word"] = df["body_clean"].str.split().str.len()

    # Vectorized Subject Cleaning
    cleaned_body = (
        df["body_clean"].astype(str).str.replace(r"[\n\r\t]", " ", regex=True)
    )
    cleaned_subject = (
        df["subject"]
        .astype(str)
        .str.replace(r"^(?:(?:Re|Fw|Fwd|For)\s*:\s*)+", "", case=False, regex=True)
        .str.strip()
    )
    df["subject"] = cleaned_subject
    separators = np.where(cleaned_subject.str.endswith("."), "\n", ".\n")

    df["body_clean_and_subject"] = cleaned_subject + separators + cleaned_body

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(
        name="Message", con=conn, if_exists="replace", index=False, chunksize=10000
    )

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_mid ON Message (mid)")
    conn.commit()
    conn.close()

    print(f"written table Message back to {DB_PATH}")
    finish_clean_message_and_drop_folders(keep_full_history)


if __name__ == "__main__":
    test_email = """
Oliver,

I will attend.

Vince

There will be a drinks reception taking place on  monday 12 June 2000 between
6.00-7.00pm in the Lower Level of the congress  center - for speakers,
sponsors and exhibitors of Risk 2000, Boston
?
Please let me know if you would like to attend so we can guage  numbers.
?
Best regards,
Oliver
?
?
?
Direct: +44 (0)20 7484 9880
?
Risk Publications, 28-29 Haymarket, London SW1Y  4RX
Fax: +44 (0)20 7484 9800? Email: oliver@risk.co.uk
www.riskpublications.com




 - fortune.jpg
 - bombtech.jpg
 - airolane.jpg
 - watchp.jpg


 ---------------------




"Oliver Bennett" <oliver@risk.co.uk> on 06/01/2000 07:11:52 AM
Please respond to "Oliver Bennett" <oliver@risk.co.uk>
To: "Young, Derek" <Derek.Young@fmr.com>, "Vince J Kaminski"
<Vince_J_Kaminski@enron.com>, "Steven E Shreve" <shreve@matt.math.cmu.edu>,
"Stephen Ross" <sross@MIT.EDU>, "Staley, Mark" <STALEY@CIBC.CA>, "Mark D
Ames" <Mark.D.Ames@marshmc.com>, "Selvaggio, Robert" <rselvaggio@ambac.com>,
<robert.harrison@gm.com>, "Ross Mayfield" <ross@ratexchange.com>, "Ritchken,
Peter" <phr@guinness.som.cwru.edu>, "Peter. N.C. Davies" <peter@askari.com>,
"Prasad Nanisetty" <prasad.nanisetty@prudential.com>, "Philipp Schoenbucher"
<P.Schonbucher@finasto.uni-bonn.de>, "Pesco, Anthony"
<anthony.pesco@csfb.com>, "Merrell Hora" <mhora@Oppenheimerfunds.com>,
"Lirtzman, Harris" <hlirtzm@comlan.cn.ci.nyc.ny.us>, "Leslie Rahl"
<LESLIE@CMRA.COM>, "John McEvoy" <john@creditex.com>, "John Hull"
<hull@mgmt.utoronto.ca>, "Joe Pimbley" <pimbley@sbcm.com>, "Jeremy Berkowitz"
<jeremy.berkowitz@frb.gov>, "Ethan Berman" <ethan.berman@riskmetrics.com>,
"Browne, Sid" <sid.browne@gs.com>, "Bob Maynard"
<BMaynard@persi.state.id.us>, "Derman, Emanuel" <emanuel.derman@gs.com>,
<edumas@bkb.com>, <tracy@lehman.com>, <eric.zz-reiner@wdr.com>,
<jlgertie@bkb.com>, <david_rowe@infinity.com>, "gene.guill"
<gene.guill@db.com>, <gleason_william@jpmorgan.com>, "Kaiser, Daniel"
<dkaiser@bofasecurities.com>, <klaus.toft@gs.com>,
<bryan.mix@ny.email.gs.com>, <holaph@tdusa.com>, <peter.zangari@gs.com>,
<atriantis@rhsmith.umd.edu>, "Neil Chriss" <neil.chriss@mindspring.com>,
<corinne.poupard-neale@iqfinancial.com>, <turnbust@cibc.ca>,
<shaheen.dil@pncbank.com>, <moore@natgas.com>, <eraab@aigtelecom.com>,
<mvalencia@arbinet.com>, <biggersk@measurisk.com>,
<jay.newberry@citicorp.com>, <michael.haubenstock@us.pwcglobal.com>,
<lars_schmidtott@swissre.com>, <francis.longstaff@anderson.ucla.edu>,
<coleman@tc.cornell.edu>, <jim@exchange.ml.com>, <kou@ieor.columbia.edu>,
<michael.ong@abnamro.com>, <mike.brosnan@occ.treas.gov>, "Adrian.B.DSilva"
<Adrian.B.DSilva@chi.frb.org>, <alex.lipton@db.com>, <landerse@genre.com>,
"Ashvin B Chhabra" <chhabra_ashvin@jpmorgan.com>,
<darryll.hendricks@ny.frb.org>, <ray.meadows@ssmb.com>, <alla.gil@ssmb.com>,
<leo_de_bever@otpp.com>, <rcuckh@gic.com.sg>, <eduard.van.gelderen@apch.nl>,
<zerolisj@brinson.com>, <jlam@owc.com>, <jane.hiscock@barra.com>, "Culp,
Christopher" <culp@chipar.com>, "Rosengarten, Jacob"
<jacob.rosengarten@gs.com>, <michelle.mccarthy@db.com>,
<erwin_martens@putnaminv.com>, <joe.mclaughlin@db.com>,
<ken.weiller@saccapital.com>, <lizeng.zhang@bankofamerica.com>,
<james.j.vinci@us.pwcglobal.com>, <ben@blackrock.com>,
<brian.Ranson@bmo.com>, <jefferid@kochind.com>, <sbramlet@utilicorp.com>,
<jean_mrha@enron.net>, <rbanaszek@sdinet.com>, <paul.ellis@credittrade.com>,
<wmiller@cfund.org>, "Gary Galante" <galante_gary@jpmorgan.com>,
<Juan.Pujadas@Us.Pwcglobal.Com>
cc:
Subject: Risk 2000 Boston - speaker reception 12 June 2000



There will be a drinks reception taking place on  monday 12 June 2000 between
6.00-7.00pm in the Lower Level of the congress  center - for speakers,
sponsors and exhibitors of Risk 2000, Boston
?
Please let me know if you would like to attend so we can guage  numbers.
?
Best regards,
Oliver
?
?
?
Direct: +44 (0)20 7484 9880
?
Risk Publications, 28-29 Haymarket, London SW1Y  4RX
Fax: +44 (0)20 7484 9800? Email: oliver@risk.co.uk
www.riskpublications.com


"Oliver Bennett" <oliver@risk.co.uk> on 06/01/2000 07:11:52 AM
To: "Young, Derek" <Derek.Young@fmr.com>, "Vince J Kaminski"
<Vince_J_Kaminski@enron.com>, "Steven E Shreve" <shreve@matt.math.cmu.edu>,
"Stephen Ross" <sross@MIT.EDU>, "Staley, Mark" <STALEY@CIBC.CA>, "Mark D
Ames" <Mark.D.Ames@marshmc.com>, "Selvaggio, Robert" <rselvaggio@ambac.com>,
<robert.harrison@gm.com>, "Ross Mayfield" <ross@ratexchange.com>, "Ritchken,
Peter" <phr@guinness.som.cwru.edu>, "Peter. N.C. Davies" <peter@askari.com>,
"Prasad Nanisetty" <prasad.nanisetty@prudential.com>, "Philipp Schoenbucher"
<P.Schonbucher@finasto.uni-bonn.de>, "Pesco, Anthony"
<anthony.pesco@csfb.com>, "Merrell Hora" <mhora@Oppenheimerfunds.com>,
"Lirtzman, Harris" <hlirtzm@comlan.cn.ci.nyc.ny.us>, "Leslie Rahl"
<LESLIE@CMRA.COM>, "John McEvoy" <john@creditex.com>, "John Hull"
<hull@mgmt.utoronto.ca>, "Joe Pimbley" <pimbley@sbcm.com>, "Jeremy Berkowitz"
<jeremy.berkowitz@frb.gov>, "Ethan Berman" <ethan.berman@riskmetrics.com>,
"Browne, Sid" <sid.browne@gs.com>, "Bob Maynard"
<BMaynard@persi.state.id.us>, "Derman, Emanuel" <emanuel.derman@gs.com>,
<edumas@bkb.com>, <tracy@lehman.com>, <eric.zz-reiner@wdr.com>,
<jlgertie@bkb.com>, <david_rowe@infinity.com>, "gene.guill"
<gene.guill@db.com>, <gleason_william@jpmorgan.com>, "Kaiser, Daniel"
<dkaiser@bofasecurities.com>, <klaus.toft@gs.com>,
<bryan.mix@ny.email.gs.com>, <holaph@tdusa.com>, <peter.zangari@gs.com>,
<atriantis@rhsmith.umd.edu>, "Neil Chriss" <neil.chriss@mindspring.com>,
<corinne.poupard-neale@iqfinancial.com>, <turnbust@cibc.ca>,
<shaheen.dil@pncbank.com>, <moore@natgas.com>, <eraab@aigtelecom.com>,
<mvalencia@arbinet.com>, <biggersk@measurisk.com>,
<jay.newberry@citicorp.com>, <michael.haubenstock@us.pwcglobal.com>,
<lars_schmidtott@swissre.com>, <francis.longstaff@anderson.ucla.edu>,
<coleman@tc.cornell.edu>, <jim@exchange.ml.com>, <kou@ieor.columbia.edu>,
<michael.ong@abnamro.com>, <mike.brosnan@occ.treas.gov>, "Adrian.B.DSilva"
<Adrian.B.DSilva@chi.frb.org>, <alex.lipton@db.com>, <landerse@genre.com>,
"Ashvin B Chhabra" <chhabra_ashvin@jpmorgan.com>,
<darryll.hendricks@ny.frb.org>, <ray.meadows@ssmb.com>, <alla.gil@ssmb.com>,
<leo_de_bever@otpp.com>, <rcuckh@gic.com.sg>, <eduard.van.gelderen@apch.nl>,
<zerolisj@brinson.com>, <jlam@owc.com>, <jane.hiscock@barra.com>, "Culp,
Christopher" <culp@chipar.com>, "Rosengarten, Jacob"
<jacob.rosengarten@gs.com>, <michelle.mccarthy@db.com>,
<erwin_martens@putnaminv.com>, <joe.mclaughlin@db.com>,
<ken.weiller@saccapital.com>, <lizeng.zhang@bankofamerica.com>,
<james.j.vinci@us.pwcglobal.com>, <ben@blackrock.com>,
<brian.Ranson@bmo.com>, <jefferid@kochind.com>, <sbramlet@utilicorp.com>,
<jean_mrha@enron.net>, <rbanaszek@sdinet.com>, <paul.ellis@credittrade.com>,
<wmiller@cfund.org>, "Gary Galante" <galante_gary@jpmorgan.com>,
<Juan.Pujadas@Us.Pwcglobal.Com>
cc:
Subject: Risk 2000 Boston - speaker reception 12 June 2000




 - fortune.jpg
 - bombtech.jpg
 - airolane.jpg
 - watchp.jpg


 ---------------------

    """

    print(clean_email_body(test_email))
