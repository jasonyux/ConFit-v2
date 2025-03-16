from datetime import datetime
import jsonlines


def parse_skills_dict_for_rater(skills):
    if not skills:
        return None
    return ', '.join(i.get('skillName') for i in skills)


def parse_location(loc_dict):
    loc_segs = []
    for seg_ in loc_dict.get('englishDisplay', loc_dict.get('originDisplay', None)).split(', '):
        if seg_ not in loc_segs:
            loc_segs.append(seg_)
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


def generate_job_doc_dict(doc: dict):
    text = doc.get('text')
    if len(doc.get('text', '')) < 20:
        return {}
    notes = doc.get('notes', [])
    job_functions = doc.get('jobFunctions', [])
    industries = doc.get('industries', None)
    preferred_languages = doc.get('preferredLanguages', [])
    required_languages = doc.get('requiredLanguages', [])
    required_skills=doc.get('requiredSkills', None)
    preferred_skills=doc.get('preferredSkills', None)
    allow_remote=doc.get('allowRemote', False)
    # parse annual_salary
    annual_salary_ = doc.get('annualSalaryInUSD', {})
    gte = annual_salary_.get('gte')
    lte = annual_salary_.get('lte')
    if gte and lte:
        annual_salary = f"from {gte} to {lte} USD per year"
    elif lte:
        annual_salary = f"less than {lte} USD per year"
    elif gte:
        annual_salary = f"more than {lte} USD per year"
    else:
        annual_salary = None
    job_type = doc.get('jobType')
    company_info= doc.get('companyInfo', {}) or {}
    # calc exp year
    exp_years_ = doc.get('experienceYearRange', {})
    if len(exp_years_) == 0:
        exp_years = None
    else:
        lte = exp_years_.get('lte', 0)
        gte = exp_years_.get('gte', 0)
        if lte and not gte:
            exp_years = f'less than {lte} years'
        elif lte and gte:
            exp_years = f"from {gte} to {lte} years"
        elif not lte and gte:
            exp_years = f"more than {gte} years"
        elif not lte and not gte:  # convention: {lte: 0} is fresh graduate 
            exp_years = 'fresh graduate'
    # calc contract length
    start_date_ = doc.get('startDate')
    end_date_ = doc.get('endDate')
    contract_length, s_date_str = None, None
    if start_date_ and end_date_:
        s_date = datetime.strptime(start_date_, '%Y-%m-%d')
        e_date = datetime.strptime(end_date_, '%Y-%m-%d')
        contract_length = f'{(e_date-s_date).days//30} months'
    elif start_date_:
        s_date_str = start_date_

    # process job notes
    job_notes_dict = {}
    for note in notes:
        if 'lastModifiedDate' in note:
            job_notes_dict[f'Note at {note["lastModifiedDate"][:10]}'] = note.get("text", "")
        else:
            job_notes_dict[f'Note'] = note.get("text", "")
    
    res = {
        'title': doc.get('title'),
        'job type': job_type,
        'openings': doc.get('openings', None),
        'minimum degree': doc.get('minimumDegreeLevel', None),
        'preferred degrees': doc.get('preferredDegreeLevels', None),
        'industries': industries,
        'job functions': job_functions,
        'job start on': s_date_str,
        'contract length': contract_length,
        'locations': list(set(loc_str for loc in doc.get('locations', []) if (loc_str:=parse_location(loc)))),
        'work remotely': allow_remote,
        'required skills': parse_skills_dict_for_rater(required_skills),
        'preferred skills': parse_skills_dict_for_rater(preferred_skills),
        'required experience': exp_years,
        'required languages': required_languages,
        'preferred languages': preferred_languages,
        'estimated salary': annual_salary,
        'company name': doc.get('companyName', company_info.get('businessName', None)),
        'company info': company_info_regulator(company_info=company_info),
        'department': doc.get('department', None),
        'sponsored work authorization': doc.get('sponsorWorkAuths', None),
        'job notes': job_notes_dict,
    }
    return dict((k,v) for k, v in res.items() if v)


def generate_job_text_dict(doc: dict):
    res = generate_job_doc_dict(doc)
    for k in ('industries', 'minimum degree', 'preferred degrees', 'required skills', 'preferred skills', 'required experience', 'required languages', 'preferred languages'):
        res.pop(k, None)

    summary_text = doc.get('summaryText', '')
    requirement_text = doc.get('requirementText', '')
    responsibility_text = doc.get('responsibilityText', '')

    # if len(summary_text + requirement_text + responsibility_text) < 1:
    #     return res
    
    res.update({
        'summaryText': summary_text,
        'requirementText': requirement_text,
        'responsibilityText': responsibility_text
    })
    # change to {'text': doc.get('text')} if you want to use raw job description but I do not think there's any difference.
    # for augments, change to {'text': augment_text }
    return dict((k,v) for k, v in res.items() if v)


# example usage of the processor funcs:
# - generate_job_doc_dict
# - generate_job_text_dict
if __name__ == "__main__":
    # test run
    job_data_path = "dataset/recruiting_data_0729/job_snapshot_240702_v2.json"

    with jsonlines.open(job_data_path) as reader:
        job_data = list(reader)
    ### process job
    all_job_parsed = []
    num_skipped = 0
    for record in job_data:
        job = generate_job_doc_dict(record["_source"]["snapShotJson"])
        if job is None or len(job) == 0:
            num_skipped += 1
            continue
        all_job_parsed.append(job)
    
    print(f"Skipped {num_skipped} jobs with generate_job_doc_dict")
    print(f"Processed {len(all_job_parsed)} jobs")

    all_job_parsed_text = []
    num_skipped = 0
    for record in job_data:
        job = generate_job_text_dict(record["_source"]["snapShotJson"])
        if job is None or len(job) == 0:
            num_skipped += 1
            continue
        all_job_parsed_text.append(job)
    
    print(f"Skipped {num_skipped} jobs with generate_job_text_dict")
    print(f"Processed {len(all_job_parsed_text)} jobs")