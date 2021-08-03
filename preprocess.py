import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
from configparser import ConfigParser


config = ConfigParser()
config.read('./config.ini')
DATA_PATH = config['DEFAULT']['DATA_PATH']
MODEL_PATH = config['DEFAULT']['MODEL_PATH']


def process_data(l):
    cols = ['Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
            'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
            'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
            'Symptom_15', 'Symptom_16', 'Symptom_17']
    # create df
    noo = pd.DataFrame(columns=cols)
    l = l.split(',')
    # make the list of size 17 only
    if len(l) >= 17:
        l = l[:17]
    else:
        l = l + [0]*(17-len(l))
    l = ['dis'] + l
    # add only one row as symp
    noo.loc[len(noo)] = l

    df = noo
    # do same pre processing
    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)

    symp = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',       'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',       'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',       'breathlessness', 'sweating', 'dehydration', 'indigestion',       'headache', 'yellowish_skin', 'dark_urine', 'nausea',
            'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
            'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
            'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
            'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
            'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',     'hip_joint_pain', 'muscle_weakness', 'stiff_neck',       'swelling_joints', 'movement_stiffness', 'spinning_movements',       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',      'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine',    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',     'altered_sensorium', 'red_spots_over_body', 'belly_pain',       'abnormal_menstruation', 'dischromic_patches',     'watering_from_eyes', 'increased_appetite', 'polyuria',       'family_history', 'mucoid_sputum', 'rusty_sputum',    'lack_of_concentration', 'visual_disturbances',       'receiving_blood_transfusion', 'receiving_unsterile_injections',       'coma', 'stomach_bleeding', 'distention_of_abdomen',       'history_of_alcohol_consumption', 'blood_in_sputum',       'prominent_veins_on_calf', 'palpitations', 'painful_walking',       'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',       'silver_like_dusting', 'small_dents_in_nails',       'inflammatory_nails', 'blister', 'red_sore_around_nose',       'yellow_crust_ooze', 'prognosis']

    vals = df.values
    symptoms = symp
    df1 = pd.read_csv(
        DATA_PATH)
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom']
                                        == symptoms[i]]['weight'].values[0]

    d = pd.DataFrame(vals, columns=cols)
    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination', 0)
    df = d.replace('foul_smell_of urine', 0)
    data = df.iloc[:, 1:].values
    return data


if __name__ == '__main__':
    process_data(DATA_PATH, MODEL_PATH)
