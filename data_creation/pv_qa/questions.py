class WeightQuestion(object):
    q = "What is the weight of the patient?"
    a = "{patientweight} kg."
    t = "Weight"

    def from_report(self, report):
        if report.patient.patientweight:
            return [
                (
                    self.q,
                    self.a.format(patientweight=report.patient.patientweight),
                    self.t,
                )
            ]
        else:
            return []


class DrugsQuestion(object):
    q = "Give an alphabetized list of all active substances of drugs taken by the patient who experienced an adverse drug reaction, that could have caused this reaction. For every drug, give the most specific active substance that is supported by the text. Answer only with a comma-separated list. Do not include generic drug classes."
    t = "DrugsGivenReaction"

    def from_report(self, report):
        # get all activesubstances
        activesubstances = []
        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                activesubstances.append(drug.activesubstance.activesubstancename)
        # deduplicate and sort
        activesubstances = ", ".join(sorted(list(set(activesubstances))))

        q_list = [self.q]
        a_list = [activesubstances]
        t_list = [self.t]

        return list(zip(q_list, a_list, t_list))


class DrugsGivenReactionQuestion(object):
    q = "Give an alphabetized list of all active substances of drugs taken by the patient who experienced '{reaction}', that could have caused this reaction. For every drug, give the most specific active substance that is supported by the text. Answer only with a comma-separated list. Do not include generic drug classes."
    t = "DrugsGivenReaction"

    def from_report(self, report):
        # get all activesubstances
        activesubstances = []
        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                activesubstances.append(drug.activesubstance.activesubstancename)
        # deduplicate and sort
        activesubstances = ", ".join(sorted(list(set(activesubstances))))

        q_list = []
        a_list = [activesubstances] * len(report.patient.reaction)
        t_list = [self.t] * len(report.patient.reaction)

        for reaction in report.patient.reaction:
            q_list.append(self.q.format(reaction=reaction.reactionmeddrapt))

        return list(zip(q_list, a_list, t_list))


class DrugIndicationQuestion(object):
    q = "What was the indication of drug {drug}?"
    t = "DrugIndication"

    def from_report(self, report):
        q_list = []
        a_list = []

        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                if drug.drugindication:
                    q_list.append(
                        self.q.format(drug=drug.activesubstance.activesubstancename)
                    )
                    a_list.append(drug.drugindication)

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class DrugAdministrationRouteQuestion(object):
    q = "What was the administration route of drug '{drug}'?"
    t = "DrugAdministrationRoute"

    administrationroute_map = {
        "001": "Auricular (otic)",
        "002": "Buccal",
        "003": "Cutaneous",
        "004": "Dental",
        "005": "Endocervical",
        "006": "Endosinusial",
        "007": "Endotracheal",
        "008": "Epidural",
        "009": "Extra-amniotic",
        "010": "Hemodialysis",
        "011": "Intra corpus cavernosum",
        "012": "Intra-amniotic",
        "013": "Intra-arterial",
        "014": "Intra-articular",
        "015": "Intra-uterine",
        "016": "Intracardiac",
        "017": "Intracavernous",
        "018": "Intracerebral",
        "019": "Intracervical",
        "020": "Intracisternal",
        "021": "Intracorneal",
        "022": "Intracoronary",
        "023": "Intradermal",
        "024": "Intradiscal (intraspinal)",
        "025": "Intrahepatic",
        "026": "Intralesional",
        "027": "Intralymphatic",
        "028": "Intramedullar (bone marrow)",
        "029": "Intrameningeal",
        "030": "Intramuscular",
        "031": "Intraocular",
        "032": "Intrapericardial",
        "033": "Intraperitoneal",
        "034": "Intrapleural",
        "035": "Intrasynovial",
        "036": "Intratumor",
        "037": "Intrathecal",
        "038": "Intrathoracic",
        "039": "Intratracheal",
        "040": "Intravenous bolus",
        "041": "Intravenous drip",
        "042": "Intravenous (not otherwise specified)",
        "043": "Intravesical",
        "044": "Iontophoresis",
        "045": "Nasal",
        "046": "Occlusive dressing technique",
        "047": "Ophthalmic",
        "048": "Oral",
        "049": "Oropharingeal",
        "050": "Other",
        "051": "Parenteral",
        "052": "Periarticular",
        "053": "Perineural",
        "054": "Rectal",
        "055": "Respiratory (inhalation)",
        "056": "Retrobulbar",
        "057": "Sunconjunctival",
        "058": "Subcutaneous",
        "059": "Subdermal",
        "060": "Sublingual",
        "061": "Topical",
        "062": "Transdermal",
        "063": "Transmammary",
        "064": "Transplacental",
        "065": "Unknown",
        "066": "Urethral",
        "067": "Vaginal",
    }

    def from_report(self, report):
        q_list = []
        a_list = []

        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                if drug.drugadministrationroute:
                    q_list.append(
                        self.q.format(drug=drug.activesubstance.activesubstancename)
                    )
                    a_list.append(
                        self.administrationroute_map[drug.drugadministrationroute]
                    )

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class DrugDosageQuestion(object):
    q = "What was the dosage of drug '{drug}'?"
    a = "{dosagenumb} {dosageunit}."
    t = "DrugDosage"

    dosage_map = {
        "001": "kg (kilograms)",
        "002": "g (grams)",
        "003": "mg (milligrams)",
        "004": "µg (micrograms)",
    }

    def from_report(self, report):
        q_list = []
        a_list = []

        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                if drug.drugstructuredosagenumb and drug.drugstructuredosageunit:
                    q_list.append(
                        self.q.format(drug=drug.activesubstance.activesubstancename)
                    )
                    a_list.append(
                        self.a.format(
                            dosagenumb=drug.drugstructuredosagenumb,
                            dosageunit=self.dosage_map[drug.drugstructuredosageunit],
                        )
                    )

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class DrugDosageTextQuestion(object):
    q = "What was the dosage of drug '{drug}'?"
    t = "DrugDosageText"

    def from_report(self, report):
        q_list = []
        a_list = []

        for drug in report.patient.drug:
            if drug.activesubstance and drug.activesubstance.activesubstancename:
                if drug.drugdosagetext:
                    q_list.append(
                        self.q.format(drug=drug.activesubstance.activesubstancename)
                    )
                    a_list.append(drug.drugdosagetext)

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class ReactionOutcomeQuestion(object):
    q = "What was the outcome of reaction '{reaction}'? Choose one of 'Recovered', 'Recovering', 'Not recovered', 'Recovered with sequelae (consequent health issues)', 'Fatal' or 'Unknown'."
    t = "ReactionOutcome"

    outcome_map = {
        "1": "Recovered",
        "2": "Recovering",
        "3": "Not recovered",
        "4": "Recovered with sequelae (consequent health issues)",
        "5": "Fatal",
        "6": "Unknown",
    }

    def from_report(self, report):
        q_list = []
        a_list = []

        for reaction in report.patient.reaction:
            if reaction.reactionoutcome:
                q_list.append(self.q.format(reaction=reaction.reactionmeddrapt))
                a_list.append(self.outcome_map[reaction.reactionoutcome])

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class PatientAgeGivenReactionQuestion(object):
    q = "What was the age group of the patient who experienced '{reaction}'?"
    t = "PatientAgeGivenReaction"

    outcome_map = {
        "1": "neonate",
        "2": "infant",
        "3": "child",
        "4": "adolescent",
        "5": "adult",
        "6": "elderly",
    }

    def from_report(self, report):
        q_list = []
        a_list = []

        if report.patient.patientagegroup:
            for reaction in report.patient.reaction:
                if reaction.reactionoutcome:
                    q_list.append(self.q.format(reaction=reaction.reactionmeddrapt))
                    a_list.append(self.outcome_map[report.patient.patientagegroup])

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))


class PatientAgeGivenDrugQuestion(object):
    q = "What was the age group of the patient who took '{drug}'? Choose one of 'neonate', 'infant', 'child', 'adolescent', 'adult', or 'elderly'."
    t = "PatientAgeGivenDrug"

    outcome_map = {
        "1": "neonate",
        "2": "infant",
        "3": "child",
        "4": "adolescent",
        "5": "adult",
        "6": "elderly",
    }

    def from_report(self, report):
        q_list = []
        a_list = []

        if report.patient.patientagegroup:
            for drug in report.patient.drug:
                if drug.activesubstance and drug.activesubstance.activesubstancename:
                    q_list.append(
                        self.q.format(drug=drug.activesubstance.activesubstancename)
                    )
                    a_list.append(self.outcome_map[report.patient.patientagegroup])

        t_list = [self.t] * len(a_list)
        return list(zip(q_list, a_list, t_list))
