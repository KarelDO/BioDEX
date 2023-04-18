import re
from string import punctuation

#### FAERS ####

# different regex functions
def faers_process_title_simple(title):
    try:
        return re.split(r"\s*[.;]+\s*", title.strip("."))[1].lower()
    except:
        return ""


def faers_process_title_advanced(title):
    # remove some confusing headers
    title = re.sub(r"(DOI|doi)([: ]+)10[.][^ ]+", "", title)
    title = re.sub(r" ET. ", " ET ", title)
    title = re.sub(r" ET.", " ET ", title)
    title = re.sub(r",? ET AL[ .]+", ". ", title, flags=re.I)
    # strip leading and trailing punctuation
    title = _strip_punctuation(title)
    # remove year at the end
    title = re.sub(r"\d{4}$", "", title)
    # strip leading and trailing punctuation
    title = _strip_punctuation(title)
    # letters to the editor
    title = re.sub(r"EDITOR.", "EDITOR", title)

    # remove the author section
    if "." in title[0:135]:
        title = re.sub(r"^([^.]|[.][^ .])+[.]+[ ]", "", title)
    else:
        # we will do our best to guess
        pass
        # title = re.sub(r'^ *[-A-Za-z\']+ [A-Z][-]?[A-Z]?[.]?(,? [-A-Za-z\']+ [A-Z][-]?[A-Z]?[.]?)*[ .]+', '', title)
    title = title.strip(". ")
    # we now start with the title, but sometimes it's difficult to know where it ends
    title = re.sub(r"[.] A (CASE|STUDY|REPORT)", r": A \1", title, flags=re.I)
    title = re.sub(r"[.] (CASE|STUDY|REPORT) OF ", r": \1 of ", title, flags=re.I)
    # remove the section after the title
    title = re.sub(r"[.?][ ].*$", "", title)
    # remove punctuation
    title = _remove_punctuation_and_lowercase(title)
    # strip end digits
    title = _strip_end_digits(title)

    # remove unk
    if title == "unk":
        title = ""

    # remove N/AP
    if title == "n ap":
        title = ""

    # # remove years
    # if re.search('^\d{4}$', title):
    #     title = ''

    if len(title) < 20:
        title = ""

    return title


#### PubMed ####


def pubmed_process_title_punctuation(title):
    if title:
        title = _remove_punctuation_and_lowercase(title)
    else:
        title = ""
    return title


#### Export the regex we use in the code ####

pubmed_process_title = pubmed_process_title_punctuation
faers_process_title = faers_process_title_advanced

#### helpers ####


def _remove_punctuation_and_lowercase(title):
    # remove any puncutation, we don't want it
    # title = re.sub(r'[^-a-zA-Z0-9]+', ' ', title)
    title = re.sub(r"[^a-zA-Z0-9]+", " ", title)
    # make lower case
    title = title.strip().lower()
    return title


def _strip_punctuation(title):
    title = title.strip(" ")
    title = title.strip(punctuation)
    title = title.strip(" ")
    return title


def _strip_end_digits(title):
    title = title.strip(" ")
    title = re.sub(r"[\d ]+$", "", title)
    title = title.strip(" ")
    return title


# To debug
if __name__ == "__main__":
    litref = "Macular edema and serous macular detachment after a standard dose of intracameral cefuroxime. Davila J.R., Mishra K., Leung L.-S. [In Process] Ophthalmic Surg. Lasers Imaging Retina 2021 52:11 (615-618)"

    processed = faers_process_title_advanced(litref)

    print(processed)
