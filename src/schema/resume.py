from dataclasses import dataclass, asdict
from src.schema.base import BaseEntity, is_entity_empty
from typing import List, Dict
from src.constants import EMPTY_DATA


@dataclass
class Education(BaseEntity):
    start_date: str = None
    end_date: str = None
    college_name_en: str = None
    college_name_cn: str = None
    qs2021_ranking: int = None
    major_name: str = None
    degree: str = None


@dataclass
class Location(BaseEntity):
    location_en: str = None
    location_cn: str = None
    country: str = None

    def __str__(self):
        return f"EN: {self.location_en}, CN: {self.location_cn}, Country: {self.country}"


@dataclass
class Resume_CompanyInfo(BaseEntity):
    employee_size: dict = None
    business_name: str = None
    industries: List[str] = None
    description: str = None
    licensing_scope: str = None
    all_locations: List[Location] = None

    def compute_company_size(self):
        employee_size = self.employee_size
        # convert to category
        lte = employee_size.get('lte', None)
        gte = employee_size.get('gte', None)
        if lte is not None and lte != EMPTY_DATA:
            if lte <= 100:
                return 'Small'
            elif lte <= 999:
                return 'Medium'
            else:
                return 'Large'
        elif gte is not None and gte != EMPTY_DATA:
            if gte < 100:
                return 'Small'
            elif gte < 999:
                return 'Medium'
            else:
                return 'Large'
        else:
            return EMPTY_DATA


    def __str__(self):
        company_size = self.compute_company_size()
        employee_size = self.employee_size

        self.employee_size = company_size
        out = super().__str__()
        self.employee_size = employee_size

        return out


@dataclass
class Experience(BaseEntity):
    title: str = None
    start_date: str = None
    end_date: str = None
    company_name: str = None
    location: str = None
    description: str = None
    company_info: Resume_CompanyInfo = None


@dataclass
class Skills(BaseEntity):
    text: str = None

    def __str__(self):
        return self.text


@dataclass
class Projects(BaseEntity):
    company_name: str = None
    project_name: str = None
    title: str = None
    start_date: str = None
    end_date: str = None
    description: str = None


@dataclass
class JobPreference(BaseEntity):
    preferredSalaryRange: str = None


@dataclass
class PersonalInfo(BaseEntity):
    nationality: str = None
    birth_place: str = None
    race: str = None


@dataclass
class ResumeMetaData(BaseEntity):
    earliest_created_date: str = None


@dataclass
class Resume:
    # old fields
    user_id: str = None
    education: List[Education] = None
    experiences: List[Experience] = None
    current_location: str = None
    preferred_locations: List[str] = None
    industry: List[str] = None
    languages: List[str] = None
    skills: Skills = None
    projects: List[Projects] = None
    # new fields
    job_preference: JobPreference = None
    personal_info: PersonalInfo = None
    metadata: ResumeMetaData = None

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
            if k in ['user_id']:
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
            if k in ['user_id', 'personal_info', 'metadata']:
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
        # for backwards compatibility with confitv1
        LIST_SEP = '\n-----\n'

        to_dict_string: dict[str, str] = {}
        for k, v in self.__dict__.items():
            if k in ['user_id', 'personal_info', 'metadata']:
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