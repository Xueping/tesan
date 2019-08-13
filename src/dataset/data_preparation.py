import glob
import pandas as pd
import json
import datetime

cols = ['DESYNPUF_ID', 'CLM_ADMSN_DT', 'CLM_UTLZTN_DAY_CNT', 'CLM_DRG_CD',
        'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3','ICD9_DGNS_CD_4','ICD9_DGNS_CD_5',
        'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8','ICD9_DGNS_CD_9','ICD9_DGNS_CD_10',
        'ICD9_PRCDR_CD_1','ICD9_PRCDR_CD_2','ICD9_PRCDR_CD_3','ICD9_PRCDR_CD_4','ICD9_PRCDR_CD_5',
        'ICD9_PRCDR_CD_6']


def processing_cms(source_path, output_file):
    #     path ='../data/DESynPUF/inpatient' # use your path
    all_files = glob.glob(source_path + "/*.csv")
    list_ = []
    for file_ in all_files:
        df = pd.read_csv(file_, index_col=None, dtype=object, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    data = frame[cols]
    unique_pats = data.DESYNPUF_ID.unique()
    grouped = data.groupby(['DESYNPUF_ID'])

    #     samples = unique_pats[range(1000)]
    patients = []
    for sub_id in unique_pats:
        patient = dict()
        patient['pid'] = sub_id
        visits = []
        acts = grouped.get_group(sub_id)
        for index, row in acts.iterrows():
            act = dict()
            admsn_dt = row['CLM_ADMSN_DT']
            day_cnt = row['CLM_UTLZTN_DAY_CNT']
            # skip if admision or discharge data is null
            if admsn_dt != admsn_dt:
                continue
            act['admsn_dt'] = admsn_dt
            act['day_cnt'] = day_cnt
            DXs = []
            for i in range(10):
                dx = row['ICD9_DGNS_CD_' + str(i + 1)]
                # if dx is not NaN
                if dx == dx:
                    DXs.append(dx)
            act['DXs'] = DXs
            CPTs = []
            for i in range(6):
                cpt = row['ICD9_PRCDR_CD_' + str(i + 1)]
                # if cpt is not NaN
                if cpt == cpt:
                    CPTs.append(cpt)
            act['CPTs'] = CPTs
            act['drg'] = row['CLM_DRG_CD']
            visits.append(act)
        patient['visits'] = visits
        patients.append(patient)

    with open(output_file, 'w') as outfile:
        json.dump(patients, outfile)

    return patients


def processing_mimic3(file_adm, file_dxx, file_txx, file_drug, file_drg, output_file):

    m_adm = pd.read_csv(file_adm, dtype={'HOSPITAL_EXPIRE_FLAG': object})
    m_dxx = pd.read_csv(file_dxx, dtype={'ICD9_CODE': object})
    m_txx = pd.read_csv(file_txx, dtype={'ICD9_CODE': object})
    m_drg = pd.read_csv(file_drg, dtype={'DRG_CODE': object})
    m_drug = pd.read_csv(file_drug, dtype={'NDC': object})

    # get total unique patients
    unique_pats = m_dxx.SUBJECT_ID.unique()

    patients = []  # store all preprocessed patients' data
    for sub_id in unique_pats:
        patient = dict()
        patient['pid'] = str(sub_id)
        pat_dxx = m_dxx[m_dxx.SUBJECT_ID == sub_id]  # get a specific patient's all data in dxx file
        uni_hadm = pat_dxx.HADM_ID.unique()  # get all unique admissions
        grouped = pat_dxx.groupby(['HADM_ID'])
        visits = []
        for hadm in uni_hadm:
            act = dict()
            adm = m_adm[(m_adm.SUBJECT_ID == sub_id) & (m_adm.HADM_ID == hadm)]
            admsn_dt = datetime.datetime.strptime(adm.ADMITTIME.values[0], "%Y-%m-%d %H:%M:%S")
            disch_dt = datetime.datetime.strptime(adm.DISCHTIME.values[0], "%Y-%m-%d %H:%M:%S")
            death_flag = adm.HOSPITAL_EXPIRE_FLAG.values[0]

            delta = disch_dt - admsn_dt
            act['admsn_dt'] = admsn_dt.strftime("%Y%m%d")
            act['day_cnt'] = str(delta.days + 1)

            codes = grouped.get_group(hadm)  # get all diagnosis codes in the adm
            DXs = []
            for index, row in codes.iterrows():
                dx = row['ICD9_CODE']
                # if dx is not NaN
                if dx == dx:
                    DXs.append(dx)

            TXs = []
            pat_txx = m_txx[(m_txx.SUBJECT_ID == sub_id) & (m_txx.HADM_ID == hadm)]
            tx_codes = pat_txx.ICD9_CODE.values  # get all procedure codes in the adm
            for code in tx_codes:
                if code == code:
                    TXs.append(code)

            drugs = []
            pat_drugs = m_drug[(m_drug.SUBJECT_ID == sub_id) & (m_drug.HADM_ID == hadm)]
            drug_codes = pat_drugs.NDC.values  # get all drug codes in the adm
            for code in drug_codes:
                if code == code and code != '0':
                    drugs.append(code)

            drgs = []
            pat_drgs = m_drg[(m_drg.SUBJECT_ID == sub_id) & (m_drg.HADM_ID == hadm)]
            drg_codes = pat_drgs.DRG_CODE.values  # get all drug codes in the adm
            for code in drg_codes:
                if code == code:
                    drgs.append(code)

            act['DXs'] = DXs
            act['CPTs'] = TXs
            act['DRGs'] = drgs
            act['Drugs'] = drugs
            act['Death'] = death_flag
            visits.append(act)
        print('patient {} is processed!'.format(sub_id))
        patient['visits'] = visits
        patients.append(patient)

    with open(output_file, 'w') as outfile:
        json.dump(patients, outfile)

    return patients


if __name__ == "__main__":

    mimic_flag = False

    if mimic_flag:
        file_adm = '../../dataset/mimic3/ADMISSIONS.csv'
        file_dxx = '../../dataset/mimic3/DIAGNOSES_ICD.csv'
        file_txx = '../../dataset/mimic3/PROCEDURES_ICD.csv'
        file_drug = '../../dataset/mimic3/PRESCRIPTIONS.csv'
        file_drg = '../../dataset/mimic3/DRGCODES.csv'

        output_file = '../../dataset/processed/patients_mimic3_full.json'

        processing_mimic3(file_adm, file_dxx, file_txx, file_drug, file_drg, output_file)

    else:
        files_path = '../../dataset/cms'
        output_file = '../../dataset/processed/patients_cms_full.json'
        processing_cms(files_path, output_file)