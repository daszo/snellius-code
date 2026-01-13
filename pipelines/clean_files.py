import sqlite3
import pandas as pd
import re
import numpy as np
from CE.utils.database import finish_clean_message_and_drop_folders


def remove_legal_disclaimer(text: str) -> str:
    """
    (USER LOGIC) Removes the specific legal disclaimer from a text string.
    """
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


def truncate_email_history(text: str) -> str:
    """
    (GEMINI LOGIC) Cuts the email text entirely at the first sign of a
    forwarded chain or history. This prevents indexing
    duplicate content.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cut_lines = []

    # Stop reading as soon as we see a Forward or Original Message delimiter.
    # Note: We replaced your 're.sub' with this 'break' logic.
    history_delimiters = [
        r"-----Original Message-----",
        r"----- Forwarded by",
        r"^\s*From:\s.+?Sent:\s.+?To:\s",  # Outlook Header quote
        r"^_+?$",  # Underscore line
        # Added specifically for your test case (Lotus Notes style):
        # Matches: "Name" <email> on 01/01/2000
        r'^".*?"\s+<.*?>\s+on\s+\d{2}/\d{2}/\d{4}',
    ]

    for line in lines:
        # If we hit a history marker, we stop reading the file entirely.
        if any(re.search(p, line, re.IGNORECASE) for p in history_delimiters):
            break
        cut_lines.append(line)

    return "\n".join(cut_lines)


def prune_signature_footer(text: str) -> str:
    """
    (GEMINI LOGIC - UPDATED) Reverse scan to eat the signature block.
    Now includes UK Postcodes and generic address indicators.
    """
    if not text:
        return ""

    lines = text.splitlines()

    clutter_patterns = [
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",  # US Phone numbers
        r"\b(fax|cell|mobile|office|w|c|h)\b",  # Phone labels
        r"\b(Street|St|Ave|Rd|Blvd|Suite|Floor|Haymarket)\b",  # Address markers (Added 'Haymarket')
        r"\b\d{5}\b",  # US Zip codes
        # --- NEW: UK Postcode Regex ---
        # Matches formats like "SW1Y 4RX", "W1 1AA", "EC1A 1BB"
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
        # --- NEW: Major Enron Office Cities (Risky generally, safe for Footer Pruning) ---
        # If a line at the very bottom contains "London" or "Houston"
        # and has NO sentence punctuation, it is an address.
        r"\b(London|Houston|New York|Calgary|Portland)\b",
        r"@enron\.com",  # Internal emails
        r"Best Regards,",
        r"Sincerely,",
        r"Thanks,",
        r"Regards,",  # Closers
    ]

    # Patterns that imply "Real Content" (Sentence endings)
    content_keepers = [r"[.?!]$", r"^(?!\s*$).{60,}$"]  # Long lines are usually content

    final_lines = []
    is_cleaning_footer = True

    for line in reversed(lines):
        stripped = line.strip()

        if not stripped:
            if not is_cleaning_footer:
                final_lines.append(line)
            continue

        # Check against patterns
        is_clutter = any(
            re.search(p, stripped, re.IGNORECASE) for p in clutter_patterns
        )
        is_keeper = any(re.search(p, stripped) for p in content_keepers)

        if is_cleaning_footer:
            # IF it looks like clutter AND it doesn't look like a sentence -> DELETE
            if is_clutter and not is_keeper:
                continue
            else:
                # We found the first real line of text. Stop deleting.
                is_cleaning_footer = False
                final_lines.append(line)
        else:
            final_lines.append(line)

    return "\n".join(reversed(final_lines)).strip()


def clean_email_body(text):
    """
    MASTER FUNCTION: Chains the user's regex cleaning with structure pruning.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # --- PHASE 1: Structural Truncation (GEMINI) ---
    # We do this FIRST to avoid processing text we are going to throw away.
    text = truncate_email_history(text)

    # --- PHASE 2: Regex Scrubbing (USER) ---
    # Adapted from your original clean_email_body

    text = remove_legal_disclaimer(text)

    # 1. Enron Disclaimer Block (User's Step 1)
    text = re.sub(r"\*{10,}[\s\S]*?\*{10,}", "", text)

    # 2. Quoted text markers (User's Step 3)
    text = re.sub(r"^[\s>]+", "", text, flags=re.MULTILINE)

    # 3. Header Blocks (User's Step 5)
    # Kept in case there is a header at the very top of the file
    header_block_pattern = r"(?:^.+? on \d{2}/\d{2}/\d{4}[\s\S]*?|^\s*(?:From|Sent|To|Cc|Please respond to):[\s\S]*?)^Subject:.*$"
    text = re.sub(header_block_pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # 4. Labeled Contact Info (User's Step 7a/7b)
    labeled_phone_pattern = r"\b(?:Fax|Direct|Tel)\s*:\s*[+\d\s().-]+"
    text = re.sub(labeled_phone_pattern, "", text, flags=re.IGNORECASE)

    labeled_email_pattern = (
        r"\bEmail\s*:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
    )
    text = re.sub(labeled_email_pattern, "", text, flags=re.IGNORECASE)

    # 5. Generic Emails/Phones (User's Step 8/9)
    generic_email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    text = re.sub(generic_email_pattern, "", text)

    generic_phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext)\.?\s*\d+)?"
    text = re.sub(generic_phone_pattern, "", text)

    # 6. Hyphens and Dots (User's Step 10/11)
    hyphen_pattern = r"(?:\s|^)-+(?=\s|$)|-{2,}"
    text = re.sub(hyphen_pattern, "", text)

    dot_word_pattern = r"\b\w+(?:\.[a-zA-Z]\w*)+\b"
    text = re.sub(dot_word_pattern, "", text)

    # 7. Question Marks (User's Logic)
    question_mark_pattern = r"^\s*\?+\s*(?:\r?\n|\r)?"
    text = re.sub(question_mark_pattern, "", text, flags=re.MULTILINE)

    # --- PHASE 3: Signature Pruning (GEMINI) ---
    # Now that the body is scrubbed, we eat the "Footer" signature
    # that is left hanging at the bottom.
    text = prune_signature_footer(text)

    # Final Whitespace Polish
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_email_bodies_pipeline(DB_PATH="data/enron.db"):
    print("starting to load")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    sql_query = "SELECT * FROM Message"

    df = pd.read_sql_query(sql_query, conn)

    conn.commit()
    conn.close()

    print(f"loaded database {DB_PATH}")

    # Apply cleaning
    df["body_clean"] = df["body"].apply(clean_email_body)
    df = df[df["body_clean"].str.strip() != ""]

    print(f"Cleaned database {DB_PATH} of shape {df.shape}")

    # 100x faster
    df["clean_length_character"] = df["body_clean"].str.len()
    df["clean_length_word"] = df["body_clean"].str.split().str.len()

    # add subject to clean body

    df_table_name = "body_clean_and_subject"
    # 1. Clean the body column first (Vectorized)
    cleaned_body = (
        df["body_clean"].astype(str).str.replace(r"[\n\r\t]", " ", regex=True)
    )

    cleaned_subject = (
        df["subject"]
        .astype(str)
        .str.replace(r"^Re:\s*", "", case=False, regex=True)
        .str.strip()
    )

    # 1. Determine the separator: If it ends with '.', use " \n", else ". \n"
    separators = np.where(cleaned_subject.str.endswith("."), "\n", ".\n")

    # 2. Concatenate strings element-wise (Vectorized)
    df[df_table_name] = cleaned_subject + separators + cleaned_body

    conn = sqlite3.connect(DB_PATH)

    # Write to new table 'similarities'
    # if_exists='replace' drops the table if it exists and creates a new one
    # if_exists='append' adds to it
    df.to_sql(
        name="Message",
        con=conn,
        if_exists="replace",
        index=False,
        chunksize=10000,  # Write in batches to save memory
    )

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_mid ON Message (mid)")
    conn.commit()

    conn.close()

    print("written table Message back to {DB_PATH}")
    finish_clean_message_and_drop_folders()


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
