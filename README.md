# BioDEX: Large-Scale Biomedical Adverse Drug Event Extraction for Real-World Pharmacovigilance.

This is the official repository for the [BioDEX paper](todo).

BioDEX is a raw resource for drug safety monitoring that bundles full-text and abstract-only PubMed papers with drug safety reports. These reports contain structured information about an Adverse Drug Events (ADEs) described in the papers, and are produced by medical experts in real-world settings.

BioDEX contains 19k full-text papers, 65k abstracts, and over 256k associated drug-safety reports.

<!-- We hope that our resource paves the way for (semi-)automated ICSR reporting systems, which one day could aid humans to perform drug safety monitoring. Additionally, we believe our task is a good resource to train and evaluate the biomedical capabilities of Large Language Models. -->

<!-- 
BioDEX is created by combining the following resources:
- [PubMed Medline](https://www.nlm.nih.gov/bsd/difference.html): Distribution of PubMed article metadata and abstracts.
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/): Distribution of full-text PubMed articles.
- [FAERS](https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers): Distribution of Individual Case Sefety Reports. -->

<!-- A preliminary write-up including initial results is included in `supplementary_material/`. A description of the dataset fields is given in `supplementary_material/BioDEX_Dataset_Card.pdf`. -->



## Installation
Create the conda environment and install the code: 

    conda create -n biodex python=3.9
    conda activate biodex
    pip install -r requirements.txt
    pip install .

## Load the raw resource
```python
import datasets

# load the raw dataset
dataset = datasets.load_dataset("FAERS-PubMed/raw_dataset")['train']

print(len(dataset)) # 65,648

# investigate an example
article = dataset[0].article
report = dataset[0].reports[0]

```

Optional, use our code to parse the raw resource into Python objects for easy manipulation
```python
import datasets
from src.utils import get_matches

# load the raw dataset
dataset = datasets.load_dataset("FAERS-PubMed/raw_dataset")['train']
dataset = get_matches(dataset)

print(len(dataset)) # 65,648

# investigate an example
article = dataset[0].article
report = dataset[0].reports[0]

print(article.title)
print(article.abstract)
print(article.fulltext)
print(article.fulltext_license)

print(report.title)
print(report.patient.patientsex)
print(report.patient.drugs[0].activesubstance.activesubstancename)

```

## Load the Report-Extraction dataset
```python
import datasets

# load the raw dataset
dataset = datasets.load_dataset("FAERS-PubMed/BioDEX-ICSR")

print(len(dataset['train']))        # 9,624
print(len(dataset['validation']))   # 2,407
print(len(dataset['test']))         # 3,628
```

## Use our Report-Extraction model
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("FAERS-PubMed/flan-t5-large-report-extraction")
tokenizer = AutoTokenizer.from_pretrained("FAERS-PubMed/flan-t5-large-report-extraction")

# TODO
input = """TODO"""
input = tokenizer.encoder(input)

output = model.predict(**input)

```

## This repository
### Analysis
Notebooks to reproduce our dataset analysis are found in `analysis/`.

## Tasks
BioDEX supports training models under different task formats. For now, we focus on the `icsr_extraction` task. More tasks are coming soon.

### ICSR-Extraction
Given the article abstract or full-text, the goal is to predict a coarse representation of the corresponding ICSR. This representation includes the seriousness of the event, the sex of the patient, an alphabetized list of drugs taken and an alphabetized list of reactions experienced. This is modelled as a sequence-to-sequence task. The dataset is on HuggingFace and can be used directly with conventional seq2seq pipelines. 

Code to train and evaluate on the ICSR-Extraction task is located in `tasks/icsr_extraction`. To train, run either of the following code-blocks depending on your Transformer architecture.

```
# train decoder-only models (e.g. GPT2)
$ cd tasks/icsr_extraction
$ python run_decoder_for_icsr_extraction.py `python ../config_to_cmd.py config_gpt2.yaml`
```
```
# train encoder-decoder models (e.g. T5)
$ cd tasks/icsr_extraction
$ python run_encdec_for_icsr_extraction.py `python ../config_to_cmd.py config_t5.yaml`
```
You can either pass command line arguments directly or save them in a config and parse the config to command line by running `` `python ../config_to_cmd.py config_t5.yaml` ``. Don't forget the backticks to first execute this command. Set the `text_column` field to either `abstract` or `fulltext_processed` to train a model on only the abstract or the entire paper respectively.

```
# evaluate your model
$ python evaluate_icsr_extraction.py ../../checkpoints/<your-model-folder>/generated_predictions.txt FAERS-PubMed/BioDEX-ICSR
```

More about our evaluation procedure in the paper.

Example input:
```
TITLE: A Case of Pancytopenia with Many Possible Causes: How Do You Tell Which is the Right One? 

ABSTRACT: Systemic lupus erythematosus (SLE) often presents with cytopenia(s); however, pancytopenia is found less commonly, requiring the consideration of possible aetiologies other than the primary disease. The authors describe the case of a female patient with a recent diagnosis of SLE admitted through the Emergency Department with fever of unknown origin and severe pancytopenia. She was medicated with prednisolone, hydroxychloroquine, azathioprine, amlodipine and sildenafil. Extensive investigation suggested azathioprine-induced myelotoxicity. However, the patient was found to have a concomitant cytomegalovirus (CMV) infection, with oral lesions, positive CMV viral load as well as the previously described haematological findings. Pancytopenia is always a diagnostic challenge, with drug-induced myelotoxicity, especially secondary to azathioprine, being a rare aetiology. This report reiterates the importance of the differential diagnosis of pancytopenia, especially in immunosuppressed patients with increased risk for opportunistic infections. The possibility of multiple aetiologies for pancytopenia in the same patient should be considered.Azathioprine-induced myelotoxicity is dose-dependent and pancytopenia is a rare form of presentation.Opportunistic infections should always be considered as a cause of cytopenias when immunosuppression is present. 

TEXT: INTRODUCTION Systemic lupus erythematosus (SLE) is a chronic inflammatory … [truncated]
```

Example output:
```
serious: yes 
patientsex: female 
drugs: AMLODIPINE BESYLATE, AZATHIOPRINE, HYDROXYCHLOROQUINE, PREDNISOLONE, SILDENAFIL 
reactions: Bone marrow toxicity, Cytomegalovirus infection, Cytomegalovirus mucocutaneous ulcer, Febrile neutropenia, Leukoplakia, Odynophagia, Oropharyngeal candidiasis, Pancytopenia, Product use issue, Red blood cell poikilocytes present, Vitamin D deficiency
```

## Dataset Creation
All our datasets are available on the HuggingFace hub. We are in the process of releasing all code to reproduce these datasets. 

| Task            | HuggingFace dataset                                                                                                         | Leaderboard   | Code to reproduce dataset         |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------|---------------|-----------------------------------|
| raw resource    | [FAERS-PubMed/raw_dataset](https://huggingface.co/datasets/FAERS-PubMed/raw_dataset)                                        | /             | `data_creation/raw` (coming soon) |
| ICSR-Extraction | [FAERS-PubMed/BioDEX-ICSR](https://huggingface.co/datasets/FAERS-PubMed/BioDEX-ICSR/viewer/FAERS-PubMed--BioDEX-ICSR/train) | (coming soon) | `data_creation/icsr_extraction`   |
| ICSR-QA         | (coming soon)                                                                                                               | (coming soon) | (coming soon)                     |

## Limitations
See section 9 of the [BioDEX paper](todo) for limitations and ethical considerations.

## Contact
Open an issue on this GitHub page or email `karel[dot]doosterlinck[at]ugent[dot].be` and preferrably include "[BioDEX]" in the subject.

## License
Coming soon.

## Citation
Coming soon.
