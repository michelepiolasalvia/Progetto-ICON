import time

import pandas as pd
from pyswip import Prolog

def create_kb() -> Prolog:
    prolog = Prolog()
    prolog.consult("facts2.pl")

    prolog.assertz("same_date(crime(C1), crime(C2)) :- crime_date(crime(C1), D), crime_date(crime(C2), D)")
    prolog.assertz("same_area(crime(C1), crime(C2)) :- crime_area(crime(C1), A), crime_area(crime(C2), A)")
    prolog.assertz("same_area_name(crime(C1), crime(C2)) :- crime_area_name(crime(C1), AN), crime_area_name(crime(C2), AN)")
    prolog.assertz("same_type(crime(C1), crime(C2)) :- crime_type(crime(C1), T), crime_type(crime(C2), T)")
    prolog.assertz("same_type_desc(crime(C1), crime(C2)) :- crime_type_desc(crime(C1), TD), crime_type_desc(crime(C2), TD)")
    prolog.assertz("same_district(crime(C1), crime(C2)) :- reporting_district(crime(C1), D), reporting_district(crime(C2), D)")
    prolog.assertz("same_code(crime(C1), crime(C2)) :- crime_code(crime(C1), CC), crime_code(crime(C2), CC)")
    prolog.assertz("same_premise(crime(C1), crime(C2)) :- premise_code(crime(C1), PC), premise_code(crime(C2), PC)")
    prolog.assertz("same_premise_desc(crime(C1), crime(C2)) :- premise_desc(crime(C1), PD), premise_desc(crime(C2), PD)")

    prolog.assertz("num_of_crimes_in_date(crime(C), N) :- findall(C1, same_date(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_area(crime(C), N) :- findall(C1, same_area(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_area_name(crime(C), N) :- findall(C1, same_area_name(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_type(crime(C), N) :- findall(C1, same_type(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_type_desc(crime(C), N) :- findall(C1, same_type_desc(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_district(crime(C), N) :- findall(C1, same_district(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_code(crime(C), N) :- findall(C1, same_code(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_premise(crime(C), N) :- findall(C1, same_premise(crime(C), crime(C1)), L), length(L, N)")
    prolog.assertz("num_of_crimes_in_premise_desc(crime(C), N) :- findall(C1, same_premise_desc(crime(C), crime(C1)), L), length(L, N)")

    prolog.assertz("(similar_victim_descent(crime(C1), crime(C2)) :- victim_descent(crime(C1), Descent1), victim_descent(crime(C2), Descent2), Descent1 = Descent2)")

    prolog.assertz("gender_diverse_victims(crime(C1), crime(C2)) :- victim_sex(crime(C1), Sex1), victim_sex(crime(C2), Sex2), Sex1 \\= Sex2")

    prolog.assertz("weapon_used_in_area(Weapon, Area) :- weapon_desc(crime(C), Weapon), crime_area(crime(C), Area)")

    prolog.assertz("high_crime_rate_area(crime(C)) :- crime_area(crime(C), A), findall(C1, crime_area(crime(C1), A), L), length(L, N), N > 100")
    prolog.assertz("victim_is_female(crime(C)) :- victim_sex(crime(C), 'F')")
    prolog.assertz("victim_is_male(crime(C)) :- victim_sex(crime(C), 'M')")


    prolog.assertz("victim_descent_common_in_weapon(crime(C)) :- victim_descent(crime(C), V), weapon_desc(crime(C), W), findall(C1, weapon_desc(crime(C1), W), L), length(L, N), N > 100")

    prolog.assertz("area_common_for_victim_descent(crime(C)) :- crime_area(crime(C), A), victim_descent(crime(C), V), findall(C1, victim_descent(crime(C1), V), L), length(L, N), N > 100")

    prolog.assertz("crime_involving_minor(crime(C)) :- victim_age(crime(C), A), A < 18")
    prolog.assertz("crime_involving_adult(crime(C)) :- victim_age(crime(C), A), A >= 18, A < 65")
    prolog.assertz("crime_involving_senior(crime(C)) :- victim_age(crime(C), A), A >= 65")

    prolog.assertz("high_night_crime_area(Area) :- findall(C, (crime_area(crime(C), Area), crime_time(crime(C), T), T >= 1800, T =< 600, num_of_crimes_in_time(crime(C), N), N > 50), CrimeList), length(CrimeList, N), N > 10")

    prolog.assertz("high_day_crime_area(Area) :- findall(C, (crime_area(crime(C), Area), crime_time(crime(C), T), T > 600, T < 1800, num_of_crimes_in_time(crime(C), N), N > 50), CrimeList), length(CrimeList, N), N > 10")




    return prolog


def calculate_features(kb, crime_id) -> dict:
    features_dict = {}

    features_dict["DR_NO"] = crime_id

    crime_id = f"crime({crime_id})"

    date_result = list(kb.query(f"crime_date({crime_id}, D)"))[0]["D"]
    month, day, year = map(int, date_result.strip('date()').split(','))
    features_dict["DATE"] = f"{month}/{day}/{year}"
    features_dict["AREA"] = int(list(kb.query(f"crime_area({crime_id}, A)"))[0]["A"])
    features_dict["AREA_NAME"] = list(kb.query(f"crime_area_name({crime_id}, AN)"))[0]["AN"].decode('utf-8')
    features_dict["PREMISE_CODE"] = int(list(kb.query(f"premise_code({crime_id}, PC)"))[0]["PC"])
    features_dict["PREMISE_DESC"] = list(kb.query(f"premise_desc({crime_id}, PD)"))[0]["PD"].decode('utf-8')

    features_dict["RPT DIST NO"] = list(kb.query(f"reporting_district({crime_id}, RD)"))[0]["RD"]
    features_dict["PART 1-2"] = list(kb.query(f"crime_type({crime_id}, C)"))[0]["C"]
    features_dict["CRM CD"] = list(kb.query(f"crime_code({crime_id}, CC)"))[0]["CC"]
    features_dict["CRM CD DESC"] = list(kb.query(f"crime_type_desc({crime_id}, CD)"))[0]["CD"].decode('utf-8')

    features_dict["NUM_CRIMES_DATE"] = list(kb.query(f"num_of_crimes_in_date({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_AREA"] = list(kb.query(f"num_of_crimes_in_area({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_AREA_NAME"] = list(kb.query(f"num_of_crimes_in_area_name({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_TYPE"] = list(kb.query(f"num_of_crimes_in_type({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_TYPE_DESC"] = list(kb.query(f"num_of_crimes_in_type_desc({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_DISTRICT"] = list(kb.query(f"num_of_crimes_in_district({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_CODE"] = list(kb.query(f"num_of_crimes_in_code({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_PREMISE"] = list(kb.query(f"num_of_crimes_in_premise({crime_id}, N)"))[0]["N"]
    features_dict["NUM_CRIMES_PREMISE_DESC"] = list(kb.query(f"num_of_crimes_in_premise_desc({crime_id}, N)"))[0]["N"]

    # Calcola il numero di crimini che coinvolgono vittime con la stessa discendenza
    features_dict["NUM_SIMILAR_VICTIM_DESCENT"] = \
    list(kb.query(f"findall(C1, similar_victim_descent({crime_id}, crime(C1)), L), length(L, N)"))[0]["N"]

    # Calcola il numero di crimini che coinvolgono vittime di sesso diverso
    features_dict["NUM_GENDER_DIVERSITY"] = \
    list(kb.query(f"findall(C1, gender_diverse_victims({crime_id}, crime(C1)), L), length(L, N)"))[0]["N"]

    # Calcola il numero di crimini che coinvolgono l'uso di armi in un'area specifica
    features_dict["NUM_WEAPON_USED_IN_AREA"] = list(kb.query(
        f"findall(C1, weapon_used_in_area(weapon_desc({crime_id}, W), crime_area({crime_id}, A)), L), length(L, N)"))[
        0]["N"]

    # Calcola il numero di crimini che coinvolgono vittime con la stessa discendenza e l'uso di una certa arma
    features_dict["NUM_VICTIM_DESCENT_COMMON_IN_WEAPON"] = \
    list(kb.query(f"findall(C1, victim_descent_common_in_weapon({crime_id}), L), length(L, N)"))[0]["N"]

    # Calcola il numero di crimini che coinvolgono vittime con la stessa discendenza in un'area specifica
    features_dict["NUM_AREA_COMMON_FOR_VICTIM_DESCENT"] = \
    list(kb.query(f"findall(C1, area_common_for_victim_descent(crime({crime_id})), L), length(L, N)"))[0]["N"]

    features_dict["IS_HIGH_CRIME_RATE_AREA"] = int(len(list(kb.query(f"high_crime_rate_area({crime_id})"))) > 0)

    # Calcola se la vittima è una donna
    features_dict["IS_VICTIM_FEMALE"] = int(len(list(kb.query(f"victim_is_female({crime_id})"))) > 0)

    # Calcola se la vittima è un uomo
    features_dict["IS_VICTIM_MALE"] = int(len(list(kb.query(f"victim_is_male({crime_id})"))) > 0)

    # Calcola se il crimine coinvolge un minore
    features_dict["IS_CRIME_INVOLVING_MINOR"] = int(len(list(kb.query(f"crime_involving_minor({crime_id})"))) > 0)

    # Calcola se il crimine coinvolge un adulto
    features_dict["IS_CRIME_INVOLVING_ADULT"] = int(len(list(kb.query(f"crime_involving_adult({crime_id})"))) > 0)

    # Calcola se il crimine coinvolge un anziano
    features_dict["IS_CRIME_INVOLVING_SENIOR"] = int(len(list(kb.query(f"crime_involving_senior({crime_id})"))) > 0)

    # Calcola se l'area ha un alto tasso di criminalità notturna
    features_dict["IS_HIGH_NIGHT_CRIME_AREA"] = int(len(list(kb.query(f"high_night_crime_area(crime_area({crime_id}, A))"))) > 0)

    # Calcola se l'area ha un alto tasso di criminalità diurna
    features_dict["IS_HIGH_DAY_CRIME_AREA"] = int(len(list(kb.query(f"high_day_crime_area(crime_area({crime_id}, A))"))) > 0)

    return features_dict


def produce_working_dataset(kb: Prolog, path: str):
    crimes_complete: pd.DataFrame = pd.read_csv("LA_crime.csv")

    # Prendi solo le prime 1000 righe del dataset
    crimes_complete = crimes_complete.head(1000)

    extracted_values_df = None
    first = True

    for i, crime_id in enumerate(crimes_complete["DR_NO"]):
        features_dict = calculate_features(kb, crime_id)
        if first:
            extracted_values_df = pd.DataFrame([features_dict])
            first = False
        else:
            extracted_values_df = pd.concat([extracted_values_df, pd.DataFrame([features_dict])], ignore_index=True)

    extracted_values_df.to_csv(path, index=False)
    print("Dataset creato con successo.")


knowledge_base = create_kb()
produce_working_dataset(knowledge_base, "working_dataset2.csv")

