from pydantic import BaseModel
from typing import List

from .report import Report

import re

string_re = re.compile(
    r"serious: (.*)\npatientsex: (.*)\ndrugs: (.*)\nreactions: (.*)\n"
)


def get_set_precision_and_recalls(l1, l2):
    s1 = set(l1)
    s2 = set(l2)

    intersect = s1.intersection(s2)

    p = len(intersect) / len(s1)
    r = len(intersect) / len(s2)

    return p, r


class Icsr(BaseModel):
    serious: str
    patientsex: str

    drugs: List[str]
    reactions: List[str]

    def score(self, gold):
        serious_sim = self.serious == gold.serious
        patientsex_sim = self.patientsex == gold.patientsex

        drugs_p, drugs_r = get_set_precision_and_recalls(self.drugs, gold.drugs)
        reactions_p, reactions_r = get_set_precision_and_recalls(
            self.reactions, gold.reactions
        )

        weights = [1 / 6, 1 / 6, 1 / 3, 1 / 3]
        precision = [serious_sim, patientsex_sim, drugs_p, reactions_p]
        recall = [serious_sim, patientsex_sim, drugs_r, reactions_r]

        precision = sum([w * p for w, p in zip(weights, precision)])
        recall = sum([w * r for w, r in zip(weights, recall)])

        if precision and recall:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    def __hash__(self):
        return hash((self.serious, self.patientsex, self.drugs, self.reactions))

    def to_string(self):
        return """serious: {}\npatientsex: {}\ndrugs: {}\nreactions: {}\n""".format(
            self.serious,
            self.patientsex,
            ", ".join(self.drugs),
            ", ".join(self.reactions),
        )

    @classmethod
    def from_string(cls, string: str):
        match = string_re.match(string)
        if match:
            serious = match.group(1)
            patientsex = match.group(2)
            drugs = match.group(3).split(", ")
            reactions = match.group(4).split(", ")
            return Icsr(
                serious=serious, patientsex=patientsex, drugs=drugs, reactions=reactions
            )
        else:
            return None

    @classmethod
    def from_report(cls, report: Report):
        # get all activesubstances
        activesubstances = []
        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                activesubstances.append(drug.activesubstance.activesubstancename)
        # deduplicate and sort
        activesubstances = sorted(list(set(activesubstances)))

        # get all reactions and outcomes
        reactions = []
        for reaction in report.patient.reaction:
            if reaction.reactionmeddrapt:
                reactions.append(reaction.reactionmeddrapt)
        # deduplicate and sort
        reactions = sorted(list(set(reactions)))

        patientsex = report.patient.patientsex
        serious = report.serious

        if serious and patientsex and reactions and activesubstances:
            return Icsr(
                serious=serious,
                patientsex=patientsex,
                reactions=reactions,
                drugs=activesubstances,
            )
        else:
            return None
