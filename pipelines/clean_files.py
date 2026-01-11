import sqlite3
import pandas as pd
import re
import numpy as np
from CE.utils.database import finish_clean_message_and_drop_folders


def remove_legal_disclaimer(text: str) -> str:
    """
    Removes the specific legal disclaimer from a text string.
    Uses regex to handle varying whitespace (newlines, spaces) between words.
    """

    # The specific text to remove
    disclaimer_text = """This e-mail message may contain legally privileged and/or confidential
    information. If you are not the intended recipient(s), or the employee
    or agent responsible for delivery of this message to the intended
    recipient(s), you are hereby notified that any dissemination,
    distribution or copying of this e-mail message is strictly prohibited.
    If you have received this message in error, please immediately notify
    the sender and delete this e-mail message from your computer."""

    # 1. Split disclaimer into individual words
    # 2. Escape regex special chars (like parens in 'recipient(s)')
    # 3. Join with \s+ to match any sequence of whitespace (space, tab, newline)
    pattern_parts = [re.escape(word) for word in disclaimer_text.split()]
    regex_pattern = r"\s*".join(pattern_parts)

    # Compile pattern with IGNORECASE to catch capitalization variations
    # The pattern matches the sequence of words regardless of how they are wrapped
    compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)

    # Replace found instances with an empty string
    cleaned_text = compiled_pattern.sub("", text)

    return cleaned_text.strip()


def clean_email_body(text):
    """
    Cleans email body by removing replies, headers, and specific legal disclaimers.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = remove_legal_disclaimer(text)

    # 1. Remove the Enron Disclaimer Block
    # Matches a block starting and ending with 10+ asterisks
    text = re.sub(r"\*{10,}[\s\S]*?\*{10,}", "", text)

    # 2. Remove "Original Message" separator lines
    # Replaces the separator with an empty string instead of splitting the text
    text = re.sub(r"-+\s*Original\s*Message\s*-+", "", text, flags=re.IGNORECASE)

    # 3. Remove Quoted text markers (>)
    # Only remove the '>' characters and leading whitespace, preserving the text content
    text = re.sub(r"^[\s>]+", "", text, flags=re.MULTILINE)

    # 5. Remove Enron/Lotus Notes Header Blocks
    # Strategy: Identify the START of a header block using two common styles:
    #   1. Lotus style: "Name <email> on 01/01/2000..."
    #   2. Standard style: "From: ...", "To: ...", or "Please respond to: ..."
    # Then consume everything ([\s\S]*?) until the end of the "Subject:" line.

    header_block_pattern = r"(?:^.+? on \d{2}/\d{2}/\d{4}[\s\S]*?|^\s*(?:From|Sent|To|Cc|Please respond to):[\s\S]*?)^Subject:.*$"

    text = re.sub(header_block_pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # 7. Remove Labeled Contact Info (Fax, Direct, Email)
    # 7a. Remove Fax/Direct/Tel labels and their numbers
    # Stops at characters that aren't digits/symbols (like '?' or letters).
    labeled_phone_pattern = r"\b(?:Fax|Direct|Tel)\s*:\s*[+\d\s().-]+"
    text = re.sub(labeled_phone_pattern, "", text, flags=re.IGNORECASE)

    # 7b. Remove "Email:" label and the specific address following it
    # We do this here to remove the label and address as a single unit.
    labeled_email_pattern = (
        r"\bEmail\s*:\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
    )
    text = re.sub(labeled_email_pattern, "", text, flags=re.IGNORECASE)

    # 8. Remove Remaining Generic Email Addresses
    # Catches any emails that didn't have the "Email:" label.
    generic_email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    text = re.sub(generic_email_pattern, "", text)

    # 9. Remove Remaining Generic Phone Numbers
    # Catches any phones that didn't have Fax/Direct labels.
    generic_phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext)\.?\s*\d+)?"
    text = re.sub(generic_phone_pattern, "", text)
    # 10. Remove Hyphen Sequences (Separators and Standalone Dashes)
    # Matches:
    #   1. Standalone hyphens surrounded by whitespace/start/end (protects "co-worker")
    #   2. Sequences of 2 or more hyphens anywhere (like "-----")
    hyphen_pattern = r"(?:\s|^)-+(?=\s|$)|-{2,}"
    text = re.sub(hyphen_pattern, "", text)

    # 11. Remove words with a dot (Modified to preserve numbers/times)
    # Matches "word.letter..." (e.g., "risk.co.uk", "file.txt")
    # Does NOT match "digit.digit" (e.g., "6.00", "3.14")
    # \.[a-zA-Z] ensures the character after the dot is a letter.
    dot_word_pattern = r"\b\w+(?:\.[a-zA-Z]\w*)+\b"
    text = re.sub(dot_word_pattern, "", text)

    # ^ matches start of line.
    # \s* matches optional indentation.
    # \?+ matches one or more literal question marks.
    # (?:\r?\n|\r)? matches the newline at the end (optional, in case it's the last line).
    question_mark_pattern = r"^\s*\?+\s*(?:\r?\n|\r)?"

    text = re.sub(question_mark_pattern, "", text, flags=re.MULTILINE)

    # 5. Clean up extra whitespace created by removals
    # Collapse multiple newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_email_bodies_pipeline(DB_PATH="enron.db"):
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
