from datetime import datetime
import jsonlines
import json
from tqdm import tqdm


def parse_location(loc_dict):
    if not loc_dict:
        return ''
    if display_ := loc_dict.get('englishDisplay', loc_dict.get('originDisplay', None)):
        loc_segs = []
        for seg_ in display_.split(', '):
            if seg_ not in loc_segs:
                loc_segs.append(seg_)
    else:
        return loc_dict['location']
    return ', '.join(loc_segs)


def employee_num_to_company_size(num):
    """
    # Micro enterprises: 1-9 employees 
    # Small enterprises: 10-19 or 20-49 employees 
    # Medium-sized enterprises: 50-249 employees 
    # Large enterprises: 250 or more employees
    """
    if num < 10:
        return 'Micro'
    elif num < 50:
        return 'Small'
    elif num <250:
        return 'Medium'
    else:
        return 'Large'


def company_info_regulator(company_info):
    result = {}
    # determine company size
    employee_range = company_info.get('employeeRange', {}) or {}  # in case it got None
    employee_range = {k: v for k, v in employee_range.items() if v is not None}
    employee_num = sorted(employee_range.values())
    company_size = list(map(employee_num_to_company_size, employee_num))
    if len(company_size) == 1 or (len(company_size) == 2 and company_size[0]==company_size[1]):
        result['company size'] = f'{company_size[0]} enterprise'
    elif len(company_size) == 2:
        result['company size'] = f'{company_size[0]} to {company_size[1]} enterprise'
    # description
    if desc := company_info.get('description'):
        result['company description'] = desc
    elif desc := company_info.get('businessLicensingScope'):
        result['company description'] = desc
    return result



import math
import itertools
DEGREES = {"PHD":95, "JD": 96, "MD": 97, "MBA": 71, "MASTER": 70, "BACHELOR": 50, "ASSOCIATE": 30, "HIGH_SCHOOL": 20, "DVM": 90, "POSTDOC": 100, "GRADUATE_ONGOING": 60, "UNDERGRADUATE_ONGOING": 40, "CERTIFICATE": 15, "ONLINE_DEGREE": 10}
EDU_ADJ = {0: 'top', 1: 'second top', 2: 'third top'}
EXP_ADJ = {0: 'most recent', 1: 'second', 2: 'third'}

def get_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        print("Incorrect data format, should be YYYY-MM-DD")


def get_working_length(exp: dict, ref_date):
    if sd_str := exp.get('startDate'):
        sd_ = get_date(sd_str)
        if ed_str := exp.get('endDate'):
            ed_ = get_date(ed_str)
            s_days_ago = (ref_date - sd_).days
            if s_days_ago < 365:
                ago_str = f' starting from {math.ceil(s_days_ago/30)} months ago'
            else:
                ago_str = f" starting from {s_days_ago//365} years ago"
        else:
            ed_ = datetime.utcnow()
            ago_str = ""
        l_days = (ed_ - sd_).days
        if l_days < 365:
            return f"{math.ceil(l_days/30)} months{ago_str}"
        else:
            return f"{l_days//365} years{ago_str}"
    return ""
    

def generate_edu_summ(edu_doc, ref_date):
    # education
    deg_str = ""
    if degree := edu_doc.get("degreeLevel", None):
        if DEGREES[degree] >= 30:  # associate
            deg_str = f" with {degree} degree"
    if college := edu_doc.get("collegeName", None):
        college_str = f" in {college}"
    else:
        college_str = ""
    if major := edu_doc.get("majorName", None):
        major_str = f" in {major}"
    else:
        major_str = ""
    # gen college info
    college_desc_str = ""
    if college_info := edu_doc.get("collegeInfo", None):
        if categories := college_info.get("categories"):
            if "985" in categories:
                cate_str = ' a China 985'
            elif "211" in categories:
                cate_str = ' a China 211'
            elif "IVYLEAGUE" in categories:  # must be with QS rank
                cate_str = ' in US ivy league and also'
            elif "ASSCOIATE_COLLEGE" in categories:
                cate_str = ' an asscoiate college'
            elif "PRIVATE" in categories:
                cate_str = ' "独立学院"'
            else:
                cate_str = ""
        else:
            cate_str = ""
        if qs_rank := college_info.get("QS2021Rank"):
            if qs_rank < 50:
                qs_str = ' in world top 50'
            elif qs_rank < 100:
                qs_str = ' in world top 100'
            elif qs_rank < 200:
                qs_str = ' in world top 200'
            elif qs_rank < 500:
                qs_str = ' in world top 500'
            else:
                qs_str = ''
        else:
            qs_str = ''
        if cate_str or qs_rank:
            college_desc_str = f" The college is{cate_str}{qs_str}."
    r_str = 'graduated' 
    if ed_str := edu_doc.get('endDate'):
        ed_ = get_date(ed_str)
        if ref_date < ed_:
            r_str = 'will graduate'
        date_str = f" in {ed_str}"
    else:
        date_str = f""
    return f'I graduated{college_str}{major_str}{deg_str}{date_str}.{college_desc_str}'


class TalentTrainingText:
    def __init__(self, doc, ref_date):
        self._doc = doc
        self._ref_date = ref_date
        self._mentioned_companies = set()
        self._base_doc = self.__gen_base_doc()
    
    def generate_talent_doc_dict(self, truncate_seg=0):
        def continue_trunc():
            return len(self.__exp_summs)+len(self.__proj_summs)+len(self.__edu_summs)-n_exp_seg-n_proj_seg-n_edu_seg < truncate_seg
            
        res = self._base_doc.copy()
        n_exp_seg, n_proj_seg, n_edu_seg = len(self.__exp_summs), len(self.__proj_summs), len(self.__edu_summs)
        if n_exp_seg+n_proj_seg+n_edu_seg >= truncate_seg+3:
            # must leave one seg for each
            while continue_trunc() and (n_exp_seg>2 or n_proj_seg>2 or n_edu_seg>2):
                if n_exp_seg>1 and continue_trunc():
                    n_exp_seg -= 1
                if n_proj_seg>1 and continue_trunc():
                    n_proj_seg -= 1
                if n_edu_seg>1 and continue_trunc() and (n_proj_seg<2 and n_exp_seg<2):
                    n_edu_seg -= 1
        
        educations = {}
        experiences = {}
        projects = {}
        for i, edu in enumerate(itertools.islice(self.__edu_summs, n_edu_seg)):
            edu_adj = EDU_ADJ.get(i, f'{i+1}th')
            educations[f'{edu_adj} education'] = edu
        for i, exp in enumerate(itertools.islice(self.__exp_summs, n_exp_seg)):
            exp_adj = EXP_ADJ.get(i, f'{i+1}th')
            experiences[f'{exp_adj} experience'] = exp
        for i, proj in enumerate(itertools.islice(self.__proj_summs, n_proj_seg)):
            proj_adj = EXP_ADJ.get(i, f'{i+1}th')
            projects[f'{proj_adj} project'] = proj
        res['educations'] = educations
        res['experiences'] = experiences
        res['projects'] = projects
        return dict((k,v) for k, v in res.items() if v)

    def generate_talent_doc_dict_desensitized(self, truncate_seg=0):
        res = self.generate_talent_doc_dict(truncate_seg)
        sensitive_keys = ['age', 'nationality', 'gender']
        for key in sensitive_keys:
            if key in res:
                res.pop(key)
        return res

    def _generate_exp_proj_summ(self, exp_proj):
        result_ = []
        if department := exp_proj.get('department'):
            depa_str = f'in the {department.replace("department", "")} department'
        else:
            depa_str = f""
        if comp_ := exp_proj.get("companyName"):
            comp_str = f"in the {comp_} "
        else:
            comp_str = ""
        if proj_ := exp_proj.get("projectName"):
            proj_str = f"for the {proj_} project "
        else:
            proj_str = ""
        if t_ := exp_proj.get("title"):
            t_str = f"as a {t_} "
        else:
            t_str = ""
        if company_loc := exp_proj.get("location"):
            l_str = f" in {company_loc}"
        else:
            l_str = ""
        if working_length := get_working_length(exp_proj, self._ref_date):
            wl_str = f'for {working_length}'
            r_str = "had worked" if 'endDate' in exp_proj else "am currently working"
        else:
            wl_str = ''
            r_str = 'worked'
        result_.append(f'I {r_str} {t_str}{depa_str}{proj_str}{comp_str}{wl_str}{l_str}'.rstrip()+'.')
        if description := exp_proj.get("description"):
            result_.append(description.strip())
        if comp_ not in self._mentioned_companies and (company_info := company_info_regulator(exp_proj.get("companyInfo", {}))):
            self._mentioned_companies.add(comp_)
            if c_size := company_info.get('company size'):
                c_str = f'is a {c_size}.'
            else:
                c_str = ':'
            desc = company_info.get('company description', '')
            result_.append(" ".join((comp_, c_str, desc)))
        return '\n'.join(result_)

    def __gen_base_doc(self):
        def degree_sorter(edu_doc):
            if level_str := edu_doc.get('degreeLevel'):
                return DEGREES[level_str]
            else:
                return -1
        job_functions = self._doc.get('jobFunctions', [])
        industries = self._doc.get('industries', None)
        self.__exp_summs = [summary for exp_ in self._doc.get('experiences', []) if (summary:=self._generate_exp_proj_summ(exp_))]
        self.__proj_summs = [summary for proj_ in self._doc.get('projects', []) if (summary:=self._generate_exp_proj_summ(proj_))]
        educations = sorted(self._doc.get('educations', []), key=degree_sorter, reverse=True)
        self.__edu_summs = [summary for edu_doc in educations if (summary:=generate_edu_summ(edu_doc, self._ref_date))]
        current_loc = parse_location(self._doc.get("currentLocation"))
        preferred_locs = [parse_location(loc_) for loc_ in self._doc.get("preferredLocations", [])]
        if len(preferred_locs) == 1 and preferred_locs[0] == current_loc:
            preferred_locs = None

        # calculate age
        age = None
        if birth_date_str := self._doc.get("birthDate"):
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
            age = (self._ref_date - birth_date).days // 365
            if age < 12 or age > 100:  # bad data
                age = None
            else:
                age = str(age)
        elif birth_year_range := self._doc.get("birthYearRange"):
            lte = birth_year_range.get('lte')
            gte = birth_year_range.get('gte')
            if lte == gte:
                age = str(self._ref_date.year - lte)
            if gte < lte:
                age = f"from {gte} to {lte}"
            
        return {
            'nameTitle': self._doc.get('nameTitle'),
            'industries': industries,
            'job functions': job_functions,
            'current location': current_loc,
            'preferred locations': preferred_locs,
            'highest degree': educations[0].get('degreeLevel') if educations else None,
            'languages': self._doc.get('languages', []),
            # No notes in this version because it is purely from resumes. Maybe we will add the notes later.
            # "notes": "\n".join(t_ for note in self._doc.get('notes', []) if (t_:=regulate_text(note.get("text", "")))),
            "skill description": self._doc.get('skillText', None),
            # No motivation and work auth in this version because it is purely from resumes. Maybe we will add them later.
            # 'motivation': self._doc.get('motivation', None),
            # 'work authorization': self._doc.get('workAuthorization')
            'age': age,
            'nationality': self._doc.get('nationality'),
            'gender': self._doc.get('gender')
        }


# example usage of the processor funcs:
# - generate_talent_doc_dict
if __name__ == "__main__":
    # test run
    resume_data_path = "dataset/recruiting_data_0729/resumes_240702_v2.json"

    with jsonlines.open(resume_data_path) as reader:
        resume_data = list(reader)

    ### process resume
    all_resume_parsed = []
    num_no_project = 0
    num_no_content = 0
    for record_ in tqdm(resume_data):
        source_ = record_["_source"]["resumeJson"]
        # if 'projects' not in source_:
        #     num_no_project += 1
            # continue
        ref_date = datetime.strptime(record_["_source"]["earliestResumeCreatedDate"], '%Y-%m-%dT%H:%M:%SZ')
        t_ = TalentTrainingText(source_, ref_date)
        resume = t_.generate_talent_doc_dict(truncate_seg=0)
        if len(resume) == 0:
            num_no_content += 1
            continue
        all_resume_parsed.append(resume)

    print(f"Skipped {num_no_content=} and {num_no_project=} resume with generate_talent_doc_dict")
    print(f"Processed {len(all_resume_parsed)} resume")


    ## print an example resume
    random_resume = all_resume_parsed[20]
    print(json.dumps(random_resume, indent=4))