from dataclasses import dataclass, asdict
from typing import List, Dict
from src.constants import EMPTY_DATA
from src.schema.base import BaseEntity, is_entity_empty
from src.schema.resume import Location


@dataclass
class RequiredQualitification(BaseEntity):
    minimum_degree_level: str = None
    experience_years: str = None
    responsibilities: str = None
    skills: List[str] = None
    languages: List[str] = None


@dataclass
class PreferredQualitification(BaseEntity):
    skills: List[str] = None
    languages: List[str] = None
    degree_levels: List[str] = None


@dataclass
class Job_CompanyInfo(BaseEntity):
    name_en: str = None
    name_cn: str = None
    employee_size: Dict[str, int] = None
    business_name: str = None
    industries: List[str] = None
    description: str = None
    licensing_scope: str = None
    all_locations: List[Location] = None


@dataclass
class JobNotes(BaseEntity):
    text: str = None
    last_modified_date: str = None
    

@dataclass
class JobLocation(BaseEntity):
    text_en: str = None
    text_cn: str = None
    country: str = None


@dataclass
class JobMetaData(BaseEntity):
    created_date: str = None
    posting_time: str = None
    last_modified_date: str = None
    openings: int = None
    max_submissions: int = None
    last_activity_time: str = None
    status: str = None



@dataclass
class Job:
    job_id: str = None
    company_info: Job_CompanyInfo = None
    title: str = None
    level: str = None
    job_type: str = None
    annual_salary: str = None
    allow_remote: bool = None
    start_end_date: str = None
    locations: List[JobLocation] = None
    required_qualifications: RequiredQualitification = None
    preferred_qualifications: PreferredQualitification = None
    job_notes: List[JobNotes] = None
    job_functions: List[str] = None
    metadata: JobMetaData = None

    def __post_init__(self):
        # make sure none of the fielsd are empty
        for k, v in self.__dict__.items():
            if v is None:
                raise Exception(f'{k} is None')
        return

    def to_dict(self):
        dict_resume = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                simple_dict_repr = []
                for item in v:
                    simple_dict_repr.append(item.dict())
                dict_resume[k] = simple_dict_repr
            elif isinstance(v, (str, int, float)):
                dict_resume[k] = v
            else:
                dict_resume[k] = v.dict()
        return dict_resume

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

    def __str__(self):
        SECTION_SEP = '## '
        LIST_SEP = '\n-----\n'

        to_string = ''
        for k, v in self.__dict__.items():
            if k in ['job_id']:
                continue

            data_str = ''
            if is_entity_empty(v):
                continue
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, BaseEntity):
                        data_str += f'{str(vv)}{LIST_SEP}'
                    else:
                        data_str += f'- {str(vv)}\n'
            else:
                data_str += str(v)
            data_str = data_str.strip()
            to_string += f"{SECTION_SEP}{k}\n{data_str}\n\n"
        to_string = to_string.strip()
        return to_string

    def desensitized_str(self) -> str:
        SECTION_SEP = '## '
        LIST_SEP = '\n-----\n'

        to_string = ''
        for k, v in self.__dict__.items():
            if k in ['job_id', 'metadata']:
                continue

            data_str = ''
            if is_entity_empty(v):
                continue
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, BaseEntity):
                        data_str += f'{str(vv)}{LIST_SEP}'
                    else:
                        data_str += f'- {str(vv)}\n'
            else:
                data_str += str(v)
            data_str = data_str.strip()
            to_string += f"{SECTION_SEP}{k}\n{data_str}\n\n"
        to_string = to_string.strip()
        return to_string

    def desensitized_str_for_confit_v1(self) -> Dict[str, str]:
        LIST_SEP = '\n-----\n'

        to_dict_string: dict[str, str] = {}
        for k, v in self.__dict__.items():
            if k in ['job_id', 'metadata']:
                continue

            data_str = ''
            if is_entity_empty(v):
                to_dict_string[k] = EMPTY_DATA
                continue
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, BaseEntity):
                        data_str += f'{str(vv)}{LIST_SEP}'
                    else:
                        data_str += f'- {str(vv)}\n'
            else:
                data_str += str(v)
            data_str = data_str.strip()
            to_dict_string[k] = data_str
        return to_dict_string