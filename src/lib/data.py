from dataclasses import dataclass
from typing import List, Dict, Any
@dataclass
class Data:
    name: str # Patient name 
    input_dir: str # Path to input directory
    output_dir: str # Path to output directory

@dataclass
class Form:
    user_name: str
    user_date_of_birth: str
    user_email: str
    user_phone: str
    user_address: str
    user_city: str
    user_state: str
    user_zip: str
    user_country: str
    
    doses: List[Dict[str, Any]]  # {"name": str, "frequency": str, "dosage": str}
    
    primary_care_physician_name: str
    primary_care_physician_phone: str
    primary_care_physician_email: str
    primary_care_physician_address: str
    primary_care_physician_city: str
    primary_care_physician_state: str
    primary_care_physician_zip: str
    primary_care_physician_country: str
    primary_care_physician_phone: str
    primary_care_physician_npi: str

    insurance_provider: str
    insurance_policy_number: str
    insurance_group_number: str
    insurance_plan_name: str

    
