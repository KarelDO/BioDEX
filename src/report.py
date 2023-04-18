from typing import List, Optional
from pydantic import BaseModel, validator

from xml.etree import ElementTree as ET
import xmltodict

# from linking.utils.preprocessing_regex import faers_process_title
from .preprocessing_regex import faers_process_title


class ActiveSubstance(BaseModel):
    activesubstancename: Optional[str] = None


class DrugRecurrence(BaseModel):
    drugrecuraction: Optional[str] = None


class Drug(BaseModel):
    actiondrug: Optional[str] = None
    activesubstance: Optional[ActiveSubstance] = None
    drugadditional: Optional[str] = None
    drugadministrationroute: Optional[str] = None
    drugauthorizationnumb: Optional[str] = None
    drugbatchnumb: Optional[str] = None
    drugcharacterization: Optional[str] = None
    drugcumulativedosagenumb: Optional[str] = None
    drugcumulativedosageunit: Optional[str] = None
    drugdosageform: Optional[str] = None
    drugdosagetext: Optional[str] = None
    drugenddate: Optional[str] = None
    drugenddateformat: Optional[str] = None
    drugindication: Optional[str] = None
    drugintervaldosagedefinition: Optional[str] = None
    drugintervaldosageunitnumb: Optional[str] = None
    drugrecurreadministration: Optional[str] = None
    drugseparatedosagenumb: Optional[str] = None
    drugstartdate: Optional[str] = None
    drugstartdateformat: Optional[str] = None
    drugstructuredosagenumb: Optional[str] = None
    drugstructuredosageunit: Optional[str] = None
    drugtreatmentduration: Optional[str] = None
    drugtreatmentdurationunit: Optional[str] = None
    medicinalproduct: Optional[str] = None
    drugrecurrence: Optional[List[DrugRecurrence]] = None


class PatientSummary(BaseModel):
    narrativeincludeclinical: Optional[str] = None


class Reaction(BaseModel):
    reactionmeddrapt: Optional[str] = None
    reactionmeddraversionpt: Optional[str] = None
    reactionoutcome: Optional[str] = None


class Patient(BaseModel):
    patientagegroup: Optional[str] = None
    patientonsetage: Optional[str] = None
    patientonsetageunit: Optional[str] = None
    patientsex: Optional[str] = None
    patientweight: Optional[str] = None
    drug: Optional[List[Drug]] = None
    reaction: Optional[List[Reaction]] = None
    summary: Optional[PatientSummary] = None


class PrimarySource(BaseModel):
    reportercountry: Optional[str] = None
    qualification: Optional[str] = None
    literaturereference: Optional[str] = None
    literaturereference_normalized: Optional[str] = None

    @validator("literaturereference_normalized", pre=True, always=True)
    def process_literaturereference(cls, v, values):
        # do nothing if we parsed a normalized refernece
        if v:
            return v
        # try to create a normalized reference
        literaturereference = values.get("literaturereference")
        if literaturereference:
            normalized = faers_process_title(literaturereference)
            if normalized != "":
                return normalized
            else:
                return None
        else:
            return None


class Sender(BaseModel):
    sendertype: Optional[str] = None
    senderorganization: Optional[str] = None


class Receiver(BaseModel):
    receivertype: Optional[str] = None
    receiverorganization: Optional[str] = None


class Report(BaseModel):
    safetyreportid: int
    safetyreportversion: int
    occurcountry: Optional[str] = None
    primarysourcecountry: Optional[str] = None
    fulfillexpeditecriteria: Optional[str] = None
    companynumb: Optional[str] = None
    primarysource: Optional[PrimarySource] = None
    sender: Optional[Sender] = None
    receiver: Optional[Receiver] = None
    reporttype: Optional[str] = None
    receivedate: Optional[str] = None
    receiptdate: Optional[str] = None
    patient: Optional[Patient] = None
    transmissiondate: Optional[str] = None
    serious: Optional[int] = None
    seriousnesscongenitalanomali: Optional[int] = None
    seriousnessdeath: Optional[int] = None
    seriousnessdisabling: Optional[int] = None
    seriousnesshospitalization: Optional[int] = None
    seriousnesslifethreatening: Optional[int] = None
    seriousnessother: Optional[int] = None


def enforce_lists(dct):
    if dct["patient"]["reaction"] and type(dct["patient"]["reaction"]) != list:
        dct["patient"]["reaction"] = [dct["patient"]["reaction"]]
    if dct["patient"]["drug"] and type(dct["patient"]["drug"]) != list:
        dct["patient"]["drug"] = [dct["patient"]["drug"]]
    for index, drug in enumerate(dct["patient"]["drug"]):
        if "drugrecurrence" in drug and type(drug["drugrecurrence"]) != list:
            dct["patient"]["drug"][index]["drugrecurrence"] = [
                dct["patient"]["drug"][index]["drugrecurrence"]
            ]
    return dct


def reports_from_file(file):
    reports = []
    with open(file, "r") as fp:
        report = None
        for line in fp.readlines():
            if "<safetyreport>" in line:
                report = line
            elif report:
                report += line
            if "</safetyreport>" in line:
                xml_dict = xmltodict.parse(report)["safetyreport"]
                xml_dict = enforce_lists(xml_dict)
                reports.append(Report.parse_obj(xml_dict))
                report = None
    return reports


if __name__ == "__main__":
    reports = []
    from tqdm import tqdm

    # with open('linking/faers/xml/ADR12Q4.xml', 'r') as fp:
    with open("linking/faers/xml/ADR16Q2.xml", "r") as fp:
        # with open('linking/faers/xml/3_ADR22Q2.xml', 'r') as fp:
        report = None
        for line in tqdm(fp.readlines()):
            if "<safetyreport>" in line:
                report = line
            elif report:
                report += line
            if "</safetyreport>" in line:
                xml_dict = xmltodict.parse(report)["safetyreport"]
                xml_dict = enforce_lists(xml_dict)
                reports.append(Report.parse_obj(xml_dict))
                report = None
