
# coding: utf-8

# # All libraries and reading data

# In[1]:


#All libraries

import pandas as pd
import numpy as np
import importlib
from importlib import reload  

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
label_encoder=preprocessing.LabelEncoder()
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns=200


# In[2]:


#Read data
crashes = pd.read_csv("Traffic_Crashes.csv",low_memory=False)
vehicle=pd.read_csv('Traffic_Vehicles.csv', low_memory=False)
people=pd.read_csv('Traffic_People.csv', low_memory=False)


# # Clean data - Drop columns, rows

# In[3]:


#drop  in all 3
#crsh
crsh=crashes
crsh=crsh.drop(['HIT_AND_RUN_I', 'BEAT_OF_OCCURRENCE','PHOTOS_TAKEN_I','INJURIES_INCAPACITATING',
              'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION', 
              'INJURIES_UNKNOWN', 'CRASH_DATE_EST_I','REPORT_TYPE',
              'CRASH_TYPE','DAMAGE','DATE_POLICE_NOTIFIED','STREET_NO', 'STREET_DIRECTION',
                'STREET_NAME','STATEMENTS_TAKEN_I','WORK_ZONE_TYPE',
                'WORKERS_PRESENT_I','MOST_SEVERE_INJURY','INJURIES_TOTAL','INJURIES_FATAL',
                'LATITUDE','LONGITUDE','LOCATION'], axis=1)

crsh = crsh.join(crsh['CRASH_DATE'].str.split(' ', 1, expand=True).rename(columns={0:'DATE_OF_CRASH', 1:'TIME_OF_CRASH'}))
crsh.drop(['TIME_OF_CRASH', 'CRASH_DATE'], axis=1, inplace=True)
crsh['DATE_PARSED'] = pd.to_datetime(crsh['DATE_OF_CRASH'], infer_datetime_format=True)
crsh.drop(['DATE_OF_CRASH'], axis=1, inplace=True)
crsh['CRASH_YEAR']= crsh['DATE_PARSED'].dt.year
crsh.drop(['DATE_PARSED'], axis=1, inplace=True)


#vhcl
vhcl=vehicle
vhcl=vhcl.drop(['TOWED_I','FIRE_I', 'TOWED_BY', 'TOWED_TO', 
          'AREA_00_I','AREA_01_I','AREA_02_I','AREA_03_I','AREA_04_I','AREA_05_I', 'AREA_06_I',
          'AREA_07_I','AREA_08_I','AREA_09_I','AREA_10_I','AREA_11_I','AREA_12_I','AREA_99_I',
          'USDOT_NO','CCMC_NO','ILCC_NO','COMMERCIAL_SRC','CARRIER_NAME','CARRIER_STATE',
          'CARRIER_CITY', 'HAZMAT_PLACARDS_I','HAZMAT_NAME','UN_NO','HAZMAT_PRESENT_I','HAZMAT_REPORT_I',
          'HAZMAT_REPORT_NO', 'MCS_REPORT_I','MCS_REPORT_NO','HAZMAT_VIO_CAUSE_CRASH_I',
          'MCS_VIO_CAUSE_CRASH_I','IDOT_PERMIT_NO', 'WIDE_LOAD_I','TRAILER1_WIDTH','TRAILER2_WIDTH',
          'TRAILER1_LENGTH','TRAILER2_LENGTH', 'CARGO_BODY_TYPE','LOAD_TYPE',
          'HAZMAT_OUT_OF_SERVICE_I','MCS_OUT_OF_SERVICE_I','HAZMAT_CLASS',
          'OCCUPANT_CNT','NUM_PASSENGERS','LIC_PLATE_STATE',
          'TRAVEL_DIRECTION','FIRST_CONTACT_POINT','MAKE','MODEL','CRASH_DATE'],
           axis=1)


# ppl


ppl=people
ppl=ppl[[
 'PERSON_TYPE','RD_NO','VEHICLE_ID','SEX','AGE','DRIVERS_LICENSE_CLASS',
 'SAFETY_EQUIPMENT','DRIVER_ACTION','DRIVER_VISION','PHYSICAL_CONDITION','PEDPEDAL_ACTION','PEDPEDAL_VISIBILITY','PEDPEDAL_LOCATION',
 'BAC_RESULT','BAC_RESULT VALUE','CELL_PHONE_USE']].copy()
ppl=ppl.loc[ppl['PERSON_TYPE']!='PASSENGER']



# In[7]:


print('CRASH : ',crsh.shape)
print('VEHICLE : ',vhcl.shape)
print('PEOPLE : ',ppl.shape)


# In[4]:


#drop records in crsh with units>6
crsh = crsh.drop(crsh[crsh.NUM_UNITS > 6].index)


# In[5]:


# Replace nulls and unknown/na with unknwn-
crsh_null=crsh.columns[crsh.isnull().any()].tolist()
vhcl_null=vhcl.columns[vhcl.isnull().any()].tolist()
ppl_null=ppl.columns[ppl.isnull().any()].tolist()
print(crsh_null)
print(vhcl_null)
print(ppl_null)


# In[12]:


# Null values in crsh
import ast

crsh['LANE_CNT'] = crsh['LANE_CNT'].str.replace(',', '')

crsh["INTERSECTION_RELATED_I"].fillna(  'UNKNWN', inplace = True) 
crsh["LANE_CNT"].fillna(0, inplace = True) 
crsh["NOT_RIGHT_OF_WAY_I"].fillna('UNKNWN', inplace = True) 
crsh["DOORING_I"].fillna('UNKNWN', inplace = True) 
crsh["WORK_ZONE_I"].fillna('UNKNWN', inplace = True) 
crsh["NUM_UNITS"].fillna(0, inplace = True) 
# Unknwn in vehicles
vhcl.VEHICLE_DEFECT.replace(['UNKNOWN'], ['UNKNWN'], inplace=True)
vhcl.VEHICLE_TYPE.replace(['UNKNOWN/NA'], ['UNKNWN'], inplace=True)
vhcl.VEHICLE_USE.replace(['UNKNOWN/NA'], ['UNKNWN'], inplace=True)
vhcl.MANEUVER.replace(['UNKNOWN/NA'], ['UNKNWN'], inplace=True)


# Null values in vhcl
vhcl["UNIT_TYPE"].fillna('UNKNWN', inplace = True) 
vhcl["VEHICLE_ID"].fillna(0, inplace = True) 
vhcl["CMRC_VEH_I"].fillna('UNKNWN', inplace = True) 

vhcl["VEHICLE_DEFECT"].fillna('UNKNWN', inplace = True) 
vhcl["VEHICLE_TYPE"].fillna('UNKNWN', inplace = True) 
vhcl["VEHICLE_USE"].fillna('UNKNWN', inplace = True) 
vhcl["MANEUVER"].fillna('UNKNWN', inplace = True) 
vhcl["EXCEED_SPEED_LIMIT_I"].fillna('UNKNWN', inplace = True) 
vhcl["CMV_ID"].fillna(0, inplace = True) 
vhcl["GVWR"].fillna(0, inplace = True) 
vhcl["TOTAL_VEHICLE_LENGTH"].fillna(0, inplace = True) 
vhcl["AXLE_CNT"].fillna(0, inplace = True) 
vhcl["VEHICLE_CONFIG"].fillna('UNKNWN', inplace = True) 

ppl.SEX.replace(['U'], ['UNKNWN'], inplace=True)
ppl.SAFETY_EQUIPMENT.replace(['USAGE UNKNOWN'], ['UNKNWN'], inplace=True)
ppl.DRIVER_ACTION.replace(['UNKNOWN'], ['UNKNWN'], inplace=True)
ppl.DRIVER_VISION.replace(['UNKNOWN'], ['UNKNWN'], inplace=True)
ppl.PHYSICAL_CONDITION.replace(['UNKNOWN'], ['UNKNWN'], inplace=True)
ppl.PEDPEDAL_LOCATION.replace(['UNKNOWN/NA'], ['UNKNWN'], inplace=True)



ppl["VEHICLE_ID"].fillna(0, inplace = True) 
ppl["SEX"].fillna('UNKNWN', inplace = True) 
ppl['AGE'].fillna(27, inplace = True) 
ppl["DRIVERS_LICENSE_CLASS"].fillna('UNKNWN', inplace = True) 
ppl["SAFETY_EQUIPMENT"].fillna('UNKNWN', inplace = True) 
ppl["DRIVER_ACTION"].fillna('UNKNWN', inplace = True) 
ppl["DRIVER_VISION"].fillna('UNKNWN', inplace = True) 
ppl["PHYSICAL_CONDITION"].fillna('UNKNWN', inplace = True) 
ppl["PEDPEDAL_ACTION"].fillna('UNKNWN', inplace = True) 
ppl["PEDPEDAL_VISIBILITY"].fillna('UNKNWN', inplace = True) 
ppl["PEDPEDAL_LOCATION"].fillna('UNKNWN', inplace = True) 
ppl['BAC_RESULT VALUE'].fillna(0, inplace = True) 
ppl["CELL_PHONE_USE"].fillna('UNKNWN', inplace = True) 


#Handle vehicle_year


vhcl.loc[vhcl['UNIT_TYPE'] == 'PEDESTRIAN', 'VEHICLE_YEAR'] = 0
vhcl.loc[vhcl['UNIT_TYPE'] == 'EQUESTRIAN', 'VEHICLE_YEAR'] = 0
vhcl.loc[vhcl['UNIT_TYPE'] == 'BICYCLE', 'VEHICLE_YEAR'] = 0
vhcl.loc[vhcl['UNIT_TYPE'] == 'UNKNWN', 'VEHICLE_YEAR'] = 0
vhcl.loc[vhcl['VEHICLE_YEAR'] >2018, 'VEHICLE_YEAR'] = 2013


vhcl = pd.merge(vhcl,crsh[['RD_NO','CRASH_YEAR']], how='left', on=['RD_NO'])
vhcl['CRASH_YEAR_NEW']=vhcl["CRASH_YEAR"]-5
vhcl.VEHICLE_YEAR.fillna(vhcl.CRASH_YEAR_NEW, inplace=True)
vhcl.VEHICLE_YEAR.fillna(2013, inplace=True)


# In[14]:


vhcl=vhcl.drop(['CRASH_YEAR', 'CRASH_YEAR_NEW'], axis=1)


crsh_null=crsh.columns[crsh.isnull().any()].tolist()
vhcl_null=vhcl.columns[vhcl.isnull().any()].tolist()
ppl_null=ppl.columns[ppl.isnull().any()].tolist()
print(crsh_null)
print(vhcl_null)
print(ppl_null)


# # Merging

# In[6]:


ppl_vhcl_id_not_null = ppl.loc[people.VEHICLE_ID.notnull()] 
ppl_vhcl_id_null = ppl.loc[people.VEHICLE_ID.isnull()] 

vp=pd.merge(vhcl,ppl_vhcl_id_not_null, how='left', on=['RD_NO','VEHICLE_ID'])


# In[7]:


vp1 = vp.loc[vp['UNIT_NO']==1] 
vp2 = vp.loc[vp['UNIT_NO']==2] 
vp3 = vp.loc[vp['UNIT_NO']==3] 
vp4 = vp.loc[vp['UNIT_NO']==4] 
vp5 = vp.loc[vp['UNIT_NO']==5] 
vp6 = vp.loc[vp['UNIT_NO']==6] 


vp1.columns =  ['CRASH_UNIT_ID_1', 'RD_NO', 'UNIT_NO_1', 'UNIT_TYPE_1', 'VEHICLE_ID_1', 'CMRC_VEH_I_1', 'VEHICLE_YEAR_1', 'VEHICLE_DEFECT_1', 'VEHICLE_TYPE_1', 'VEHICLE_USE_1', 'MANEUVER_1', 'EXCEED_SPEED_LIMIT_I_1', 'CMV_ID_1', 'GVWR_1', 'TOTAL_VEHICLE_LENGTH_1', 'AXLE_CNT_1', 'VEHICLE_CONFIG_1', 'PERSON_TYPE_1', 'SEX_1', 'AGE_1', 'DRIVERS_LICENSE_CLASS_1', 'SAFETY_EQUIPMENT_1', 'DRIVER_ACTION_1', 'DRIVER_VISION_1', 'PHYSICAL_CONDITION_1', 'PEDPEDAL_ACTION_1', 'PEDPEDAL_VISIBILITY_1', 'PEDPEDAL_LOCATION_1', 'BAC_RESULT_1', 'BAC_RESULT_VALUE_1', 'CELL_PHONE_USE_1']
vp2.columns =  ['CRASH_UNIT_ID_2', 'RD_NO', 'UNIT_NO_2', 'UNIT_TYPE_2', 'VEHICLE_ID_2', 'CMRC_VEH_I_2', 'VEHICLE_YEAR_2', 'VEHICLE_DEFECT_2', 'VEHICLE_TYPE_2', 'VEHICLE_USE_2', 'MANEUVER_2', 'EXCEED_SPEED_LIMIT_I_2', 'CMV_ID_2', 'GVWR_2', 'TOTAL_VEHICLE_LENGTH_2', 'AXLE_CNT_2', 'VEHICLE_CONFIG_2', 'PERSON_TYPE_2', 'SEX_2', 'AGE_2', 'DRIVERS_LICENSE_CLASS_2', 'SAFETY_EQUIPMENT_2', 'DRIVER_ACTION_2', 'DRIVER_VISION_2', 'PHYSICAL_CONDITION_2', 'PEDPEDAL_ACTION_2', 'PEDPEDAL_VISIBILITY_2', 'PEDPEDAL_LOCATION_2', 'BAC_RESULT_2', 'BAC_RESULT_VALUE_2', 'CELL_PHONE_USE_2']
vp3.columns =  ['CRASH_UNIT_ID_3', 'RD_NO', 'UNIT_NO_3', 'UNIT_TYPE_3', 'VEHICLE_ID_3', 'CMRC_VEH_I_3', 'VEHICLE_YEAR_3', 'VEHICLE_DEFECT_3', 'VEHICLE_TYPE_3', 'VEHICLE_USE_3', 'MANEUVER_3', 'EXCEED_SPEED_LIMIT_I_3', 'CMV_ID_3', 'GVWR_3', 'TOTAL_VEHICLE_LENGTH_3', 'AXLE_CNT_3', 'VEHICLE_CONFIG_3', 'PERSON_TYPE_3', 'SEX_3', 'AGE_3', 'DRIVERS_LICENSE_CLASS_3', 'SAFETY_EQUIPMENT_3', 'DRIVER_ACTION_3', 'DRIVER_VISION_3', 'PHYSICAL_CONDITION_3', 'PEDPEDAL_ACTION_3', 'PEDPEDAL_VISIBILITY_3', 'PEDPEDAL_LOCATION_3', 'BAC_RESULT_3', 'BAC_RESULT_VALUE_3', 'CELL_PHONE_USE_3']
vp4.columns =  ['CRASH_UNIT_ID_4', 'RD_NO', 'UNIT_NO_4', 'UNIT_TYPE_4', 'VEHICLE_ID_4', 'CMRC_VEH_I_4', 'VEHICLE_YEAR_4', 'VEHICLE_DEFECT_4', 'VEHICLE_TYPE_4', 'VEHICLE_USE_4', 'MANEUVER_4', 'EXCEED_SPEED_LIMIT_I_4', 'CMV_ID_4', 'GVWR_4', 'TOTAL_VEHICLE_LENGTH_4', 'AXLE_CNT_4', 'VEHICLE_CONFIG_4', 'PERSON_TYPE_4', 'SEX_4', 'AGE_4', 'DRIVERS_LICENSE_CLASS_4', 'SAFETY_EQUIPMENT_4', 'DRIVER_ACTION_4', 'DRIVER_VISION_4', 'PHYSICAL_CONDITION_4', 'PEDPEDAL_ACTION_4', 'PEDPEDAL_VISIBILITY_4', 'PEDPEDAL_LOCATION_4', 'BAC_RESULT_4', 'BAC_RESULT_VALUE_4', 'CELL_PHONE_USE_4']
vp5.columns =  ['CRASH_UNIT_ID_5', 'RD_NO', 'UNIT_NO_5', 'UNIT_TYPE_5', 'VEHICLE_ID_5', 'CMRC_VEH_I_5', 'VEHICLE_YEAR_5', 'VEHICLE_DEFECT_5', 'VEHICLE_TYPE_5', 'VEHICLE_USE_5', 'MANEUVER_5', 'EXCEED_SPEED_LIMIT_I_5', 'CMV_ID_5', 'GVWR_5', 'TOTAL_VEHICLE_LENGTH_5', 'AXLE_CNT_5', 'VEHICLE_CONFIG_5', 'PERSON_TYPE_5', 'SEX_5', 'AGE_5', 'DRIVERS_LICENSE_CLASS_5', 'SAFETY_EQUIPMENT_5', 'DRIVER_ACTION_5', 'DRIVER_VISION_5', 'PHYSICAL_CONDITION_5', 'PEDPEDAL_ACTION_5', 'PEDPEDAL_VISIBILITY_5', 'PEDPEDAL_LOCATION_5', 'BAC_RESULT_5', 'BAC_RESULT_VALUE_5', 'CELL_PHONE_USE_5']
vp6.columns =  ['CRASH_UNIT_ID_6', 'RD_NO', 'UNIT_NO_6', 'UNIT_TYPE_6', 'VEHICLE_ID_6', 'CMRC_VEH_I_6', 'VEHICLE_YEAR_6', 'VEHICLE_DEFECT_6', 'VEHICLE_TYPE_6', 'VEHICLE_USE_6', 'MANEUVER_6', 'EXCEED_SPEED_LIMIT_I_6', 'CMV_ID_6', 'GVWR_6', 'TOTAL_VEHICLE_LENGTH_6', 'AXLE_CNT_6', 'VEHICLE_CONFIG_6', 'PERSON_TYPE_6', 'SEX_6', 'AGE_6', 'DRIVERS_LICENSE_CLASS_6', 'SAFETY_EQUIPMENT_6', 'DRIVER_ACTION_6', 'DRIVER_VISION_6', 'PHYSICAL_CONDITION_6', 'PEDPEDAL_ACTION_6', 'PEDPEDAL_VISIBILITY_6', 'PEDPEDAL_LOCATION_6', 'BAC_RESULT_6', 'BAC_RESULT_VALUE_6', 'CELL_PHONE_USE_6']


vpa=vp1[['RD_NO']].copy()
vpa.columns=['RD_NO']
print('Vpa rd_no : ',vpa.shape)
vpa = pd.merge(vpa,vp1, how='left', on=['RD_NO'])
print('Vpa unit 1 : ',vpa.shape)

vpa = pd.merge(vpa,vp2, how='left', on=['RD_NO'])
print('Vpa unit 2 : ',vpa.shape)

vpa = pd.merge(vpa,vp3, how='left', on=['RD_NO'])
print('Vpa unit 3 : ',vpa.shape)

vpa = pd.merge(vpa,vp4, how='left', on=['RD_NO'])
print('Vpa unit 4 : ',vpa.shape)

vpa = pd.merge(vpa,vp5, how='left', on=['RD_NO'])
print('Vpa unit 5 : ',vpa.shape)

vpa = pd.merge(vpa,vp6, how='left', on=['RD_NO'])
print('Vpa unit 6 : ',vpa.shape)



# In[8]:


cvp = pd.merge(crsh,vpa, how='inner', on=['RD_NO'])


# In[9]:




v_na='UNKNWN'
cvp["PERSON_TYPE_1"].fillna(v_na, inplace = True) 
cvp["PERSON_TYPE_2"].fillna(v_na, inplace = True) 
cvp["PERSON_TYPE_3"].fillna(v_na, inplace = True) 
cvp["PERSON_TYPE_4"].fillna(v_na, inplace = True) 
cvp["PERSON_TYPE_5"].fillna(v_na, inplace = True) 
cvp["PERSON_TYPE_6"].fillna(v_na, inplace = True) 


cvp["SEX_1"].fillna(v_na, inplace = True) 
cvp["SEX_2"].fillna(v_na, inplace = True) 
cvp["SEX_3"].fillna(v_na, inplace = True) 
cvp["SEX_4"].fillna(v_na, inplace = True) 
cvp["SEX_5"].fillna(v_na, inplace = True) 
cvp["SEX_6"].fillna(v_na, inplace = True) 

v_na=27
cvp["AGE_1"].fillna(v_na, inplace = True) 
cvp["AGE_2"].fillna(v_na, inplace = True) 
cvp["AGE_3"].fillna(v_na, inplace = True) 
cvp["AGE_4"].fillna(v_na, inplace = True) 
cvp["AGE_5"].fillna(v_na, inplace = True) 
cvp["AGE_6"].fillna(v_na, inplace = True) 

v_na='UNKNWN'

cvp["SAFETY_EQUIPMENT_1"].fillna(v_na, inplace = True) 
cvp["SAFETY_EQUIPMENT_2"].fillna(v_na, inplace = True) 
cvp["SAFETY_EQUIPMENT_3"].fillna(v_na, inplace = True) 
cvp["SAFETY_EQUIPMENT_4"].fillna(v_na, inplace = True) 
cvp["SAFETY_EQUIPMENT_5"].fillna(v_na, inplace = True) 
cvp["SAFETY_EQUIPMENT_6"].fillna(v_na, inplace = True) 


cvp["DRIVER_ACTION_1"].fillna(v_na, inplace = True) 
cvp["DRIVER_ACTION_2"].fillna(v_na, inplace = True) 
cvp["DRIVER_ACTION_3"].fillna(v_na, inplace = True) 
cvp["DRIVER_ACTION_4"].fillna(v_na, inplace = True) 
cvp["DRIVER_ACTION_5"].fillna(v_na, inplace = True) 
cvp["DRIVER_ACTION_6"].fillna(v_na, inplace = True) 

cvp["DRIVER_VISION_1"].fillna(v_na, inplace = True) 
cvp["DRIVER_VISION_2"].fillna(v_na, inplace = True) 
cvp["DRIVER_VISION_3"].fillna(v_na, inplace = True) 
cvp["DRIVER_VISION_4"].fillna(v_na, inplace = True) 
cvp["DRIVER_VISION_5"].fillna(v_na, inplace = True) 
cvp["DRIVER_VISION_6"].fillna(v_na, inplace = True) 

cvp["PHYSICAL_CONDITION_1"].fillna(v_na, inplace = True) 
cvp["PHYSICAL_CONDITION_2"].fillna(v_na, inplace = True) 
cvp["PHYSICAL_CONDITION_3"].fillna(v_na, inplace = True) 
cvp["PHYSICAL_CONDITION_4"].fillna(v_na, inplace = True) 
cvp["PHYSICAL_CONDITION_5"].fillna(v_na, inplace = True) 
cvp["PHYSICAL_CONDITION_6"].fillna(v_na, inplace = True) 


cvp["PEDPEDAL_ACTION_1"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_ACTION_2"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_ACTION_3"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_ACTION_4"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_ACTION_5"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_ACTION_6"].fillna(v_na, inplace = True) 

cvp["PEDPEDAL_VISIBILITY_1"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_VISIBILITY_2"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_VISIBILITY_3"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_VISIBILITY_4"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_VISIBILITY_5"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_VISIBILITY_6"].fillna(v_na, inplace = True) 


cvp["PEDPEDAL_LOCATION_1"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_LOCATION_2"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_LOCATION_3"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_LOCATION_4"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_LOCATION_5"].fillna(v_na, inplace = True) 
cvp["PEDPEDAL_LOCATION_6"].fillna(v_na, inplace = True) 

cvp["BAC_RESULT_1"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_2"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_3"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_4"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_5"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_6"].fillna(v_na, inplace = True) 

v_na=0

cvp["BAC_RESULT_VALUE_1"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_VALUE_2"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_VALUE_3"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_VALUE_4"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_VALUE_5"].fillna(v_na, inplace = True) 
cvp["BAC_RESULT_VALUE_6"].fillna(v_na, inplace = True) 
v_na='UNKNWN'

cvp["CELL_PHONE_USE_1"].fillna(v_na, inplace = True) 
cvp["CELL_PHONE_USE_2"].fillna(v_na, inplace = True) 
cvp["CELL_PHONE_USE_3"].fillna(v_na, inplace = True) 
cvp["CELL_PHONE_USE_4"].fillna(v_na, inplace = True) 
cvp["CELL_PHONE_USE_5"].fillna(v_na, inplace = True) 
cvp["CELL_PHONE_USE_6"].fillna(v_na, inplace = True) 


cvp["UNIT_NO_1"].fillna(1, inplace = True) 
cvp["UNIT_NO_2"].fillna(2, inplace = True) 
cvp["UNIT_NO_3"].fillna(3, inplace = True) 
cvp["UNIT_NO_4"].fillna(4, inplace = True) 
cvp["UNIT_NO_5"].fillna(5, inplace = True) 
cvp["UNIT_NO_6"].fillna(6, inplace = True) 

cvp["UNIT_TYPE_1"].fillna(v_na, inplace = True) 
cvp["UNIT_TYPE_2"].fillna(v_na, inplace = True) 
cvp["UNIT_TYPE_3"].fillna(v_na, inplace = True) 
cvp["UNIT_TYPE_4"].fillna(v_na, inplace = True) 
cvp["UNIT_TYPE_5"].fillna(v_na, inplace = True) 
cvp["UNIT_TYPE_6"].fillna(v_na, inplace = True) 


cvp["CMRC_VEH_I_1"].fillna(v_na, inplace = True) 
cvp["CMRC_VEH_I_2"].fillna(v_na, inplace = True) 
cvp["CMRC_VEH_I_3"].fillna(v_na, inplace = True) 
cvp["CMRC_VEH_I_4"].fillna(v_na, inplace = True) 
cvp["CMRC_VEH_I_5"].fillna(v_na, inplace = True) 
cvp["CMRC_VEH_I_6"].fillna(v_na, inplace = True) 

v_na=2013
cvp["VEHICLE_YEAR_1"].fillna(v_na, inplace = True) 
cvp["VEHICLE_YEAR_2"].fillna(v_na, inplace = True) 
cvp["VEHICLE_YEAR_3"].fillna(v_na, inplace = True) 
cvp["VEHICLE_YEAR_4"].fillna(v_na, inplace = True) 
cvp["VEHICLE_YEAR_5"].fillna(v_na, inplace = True) 
cvp["VEHICLE_YEAR_6"].fillna(v_na, inplace = True) 

v_na='UNKNWN'
cvp["VEHICLE_DEFECT_1"].fillna(v_na, inplace = True) 
cvp["VEHICLE_DEFECT_2"].fillna(v_na, inplace = True) 
cvp["VEHICLE_DEFECT_3"].fillna(v_na, inplace = True) 
cvp["VEHICLE_DEFECT_4"].fillna(v_na, inplace = True) 
cvp["VEHICLE_DEFECT_5"].fillna(v_na, inplace = True) 
cvp["VEHICLE_DEFECT_6"].fillna(v_na, inplace = True) 

cvp["VEHICLE_USE_1"].fillna(v_na, inplace = True) 
cvp["VEHICLE_USE_2"].fillna(v_na, inplace = True) 
cvp["VEHICLE_USE_3"].fillna(v_na, inplace = True) 
cvp["VEHICLE_USE_4"].fillna(v_na, inplace = True) 
cvp["VEHICLE_USE_5"].fillna(v_na, inplace = True) 
cvp["VEHICLE_USE_6"].fillna(v_na, inplace = True) 

cvp["MANEUVER_1"].fillna(v_na, inplace = True) 
cvp["MANEUVER_2"].fillna(v_na, inplace = True) 
cvp["MANEUVER_3"].fillna(v_na, inplace = True) 
cvp["MANEUVER_4"].fillna(v_na, inplace = True) 
cvp["MANEUVER_5"].fillna(v_na, inplace = True) 
cvp["MANEUVER_6"].fillna(v_na, inplace = True) 

cvp["EXCEED_SPEED_LIMIT_I_1"].fillna(v_na, inplace = True) 
cvp["EXCEED_SPEED_LIMIT_I_2"].fillna(v_na, inplace = True) 
cvp["EXCEED_SPEED_LIMIT_I_3"].fillna(v_na, inplace = True) 
cvp["EXCEED_SPEED_LIMIT_I_4"].fillna(v_na, inplace = True) 
cvp["EXCEED_SPEED_LIMIT_I_5"].fillna(v_na, inplace = True) 
cvp["EXCEED_SPEED_LIMIT_I_6"].fillna(v_na, inplace = True) 

cvp["VEHICLE_CONFIG_1"].fillna(v_na, inplace = True) 
cvp["VEHICLE_CONFIG_2"].fillna(v_na, inplace = True) 
cvp["VEHICLE_CONFIG_3"].fillna(v_na, inplace = True) 
cvp["VEHICLE_CONFIG_4"].fillna(v_na, inplace = True) 
cvp["VEHICLE_CONFIG_5"].fillna(v_na, inplace = True) 
cvp["VEHICLE_CONFIG_6"].fillna(v_na, inplace = True) 

cvp["VEHICLE_TYPE_1"].fillna(v_na, inplace = True) 
cvp["VEHICLE_TYPE_2"].fillna(v_na, inplace = True) 
cvp["VEHICLE_TYPE_3"].fillna(v_na, inplace = True) 
cvp["VEHICLE_TYPE_4"].fillna(v_na, inplace = True) 
cvp["VEHICLE_TYPE_5"].fillna(v_na, inplace = True) 
cvp["VEHICLE_TYPE_6"].fillna(v_na, inplace = True) 


# In[10]:


print('cvp with duplicates:',cvp.shape)
cvp=cvp.drop_duplicates(subset=['RD_NO'], keep=False)
print('cvp without duplicates:',cvp.shape)
#list(cvp)


# In[11]:


234641-234584


# # Dropping few more columns not required to feed algorithms

# In[12]:



cvp=cvp.drop(['RD_NO','CRASH_UNIT_ID_1','VEHICLE_ID_1','CMV_ID_1', 'GVWR_1','TOTAL_VEHICLE_LENGTH_1','DRIVERS_LICENSE_CLASS_1','AXLE_CNT_1',
               'CRASH_UNIT_ID_2','VEHICLE_ID_2','CMV_ID_2', 'GVWR_2','TOTAL_VEHICLE_LENGTH_2','DRIVERS_LICENSE_CLASS_2','AXLE_CNT_2',
'CRASH_UNIT_ID_3','VEHICLE_ID_3','CMV_ID_3', 'GVWR_3','TOTAL_VEHICLE_LENGTH_3','DRIVERS_LICENSE_CLASS_3','AXLE_CNT_3',
'CRASH_UNIT_ID_4','VEHICLE_ID_4','CMV_ID_4', 'GVWR_4','TOTAL_VEHICLE_LENGTH_4','DRIVERS_LICENSE_CLASS_4','AXLE_CNT_4',
'CRASH_UNIT_ID_5','VEHICLE_ID_5','CMV_ID_5', 'GVWR_5','TOTAL_VEHICLE_LENGTH_5','DRIVERS_LICENSE_CLASS_5','AXLE_CNT_5',
'CRASH_UNIT_ID_6','VEHICLE_ID_6','CMV_ID_6', 'GVWR_6','TOTAL_VEHICLE_LENGTH_6','DRIVERS_LICENSE_CLASS_6','AXLE_CNT_6',
], 1)


# In[13]:


cvp_null=cvp.columns[cvp.isnull().any()].tolist()
cvp_null


# In[14]:


type(cvp)


# # One hot encoding

# In[15]:


#get dummies for crashes columns
cvp_ohe= pd.get_dummies(data=cvp, columns=['TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION',
                                'LIGHTING_CONDITION',  'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT',
                                  'ROADWAY_SURFACE_COND',  'ROAD_DEFECT', 'INTERSECTION_RELATED_I',  'NOT_RIGHT_OF_WAY_I',
                                    'DOORING_I', 'WORK_ZONE_I',
                                    'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH', 'CRASH_YEAR'])

print("Dataframe size after adding dummies to crashes columns: ", cvp_ohe.shape)


#get dummies for vehicles columns
cvp_ohe = pd.get_dummies(data=cvp_ohe, columns=[
'UNIT_TYPE_1','CMRC_VEH_I_1','VEHICLE_DEFECT_1','VEHICLE_TYPE_1','VEHICLE_USE_1', 'MANEUVER_1','EXCEED_SPEED_LIMIT_I_1','VEHICLE_CONFIG_1',
'UNIT_TYPE_2','CMRC_VEH_I_2','VEHICLE_DEFECT_2','VEHICLE_TYPE_2','VEHICLE_USE_2', 'MANEUVER_2','EXCEED_SPEED_LIMIT_I_2','VEHICLE_CONFIG_2',
'UNIT_TYPE_3','CMRC_VEH_I_3','VEHICLE_DEFECT_3','VEHICLE_TYPE_3','VEHICLE_USE_3', 'MANEUVER_3','EXCEED_SPEED_LIMIT_I_3','VEHICLE_CONFIG_3',
'UNIT_TYPE_4','CMRC_VEH_I_4','VEHICLE_DEFECT_4','VEHICLE_TYPE_4','VEHICLE_USE_4', 'MANEUVER_4','EXCEED_SPEED_LIMIT_I_4','VEHICLE_CONFIG_4',    
'UNIT_TYPE_5','CMRC_VEH_I_5','VEHICLE_DEFECT_5','VEHICLE_TYPE_5','VEHICLE_USE_5', 'MANEUVER_5','EXCEED_SPEED_LIMIT_I_5','VEHICLE_CONFIG_5',
'UNIT_TYPE_6','CMRC_VEH_I_6','VEHICLE_DEFECT_6','VEHICLE_TYPE_6','VEHICLE_USE_6', 'MANEUVER_6','EXCEED_SPEED_LIMIT_I_6','VEHICLE_CONFIG_6'   
])

print("Dataframe size after adding dummies to vehicles columns: ", cvp_ohe.shape)

#get dummies for peoples columns
cvp_ohe = pd.get_dummies(data=cvp_ohe, columns=[
'PERSON_TYPE_1', 'SEX_1', 'SAFETY_EQUIPMENT_1', 'DRIVER_ACTION_1', 'DRIVER_VISION_1', 'PHYSICAL_CONDITION_1',
'PEDPEDAL_ACTION_1', 'PEDPEDAL_VISIBILITY_1', 'PEDPEDAL_LOCATION_1', 'BAC_RESULT_1', 'CELL_PHONE_USE_1',
'PERSON_TYPE_2', 'SEX_2', 'SAFETY_EQUIPMENT_2', 'DRIVER_ACTION_2', 'DRIVER_VISION_2', 'PHYSICAL_CONDITION_2',
'PEDPEDAL_ACTION_2', 'PEDPEDAL_VISIBILITY_2', 'PEDPEDAL_LOCATION_2', 'BAC_RESULT_2', 'CELL_PHONE_USE_2',
'PERSON_TYPE_3', 'SEX_3', 'SAFETY_EQUIPMENT_3', 'DRIVER_ACTION_3', 'DRIVER_VISION_3', 'PHYSICAL_CONDITION_3',
'PEDPEDAL_ACTION_3', 'PEDPEDAL_VISIBILITY_3', 'PEDPEDAL_LOCATION_3', 'BAC_RESULT_3', 'CELL_PHONE_USE_3',
'PERSON_TYPE_4', 'SEX_4', 'SAFETY_EQUIPMENT_4', 'DRIVER_ACTION_4', 'DRIVER_VISION_4', 'PHYSICAL_CONDITION_4',
'PEDPEDAL_ACTION_4', 'PEDPEDAL_VISIBILITY_4', 'PEDPEDAL_LOCATION_4', 'BAC_RESULT_4', 'CELL_PHONE_USE_4',
'PERSON_TYPE_5', 'SEX_5', 'SAFETY_EQUIPMENT_5', 'DRIVER_ACTION_5', 'DRIVER_VISION_5', 'PHYSICAL_CONDITION_5',
'PEDPEDAL_ACTION_5', 'PEDPEDAL_VISIBILITY_5', 'PEDPEDAL_LOCATION_5', 'BAC_RESULT_5', 'CELL_PHONE_USE_5',
'PERSON_TYPE_6', 'SEX_6', 'SAFETY_EQUIPMENT_6', 'DRIVER_ACTION_6', 'DRIVER_VISION_6', 'PHYSICAL_CONDITION_6',
'PEDPEDAL_ACTION_6', 'PEDPEDAL_VISIBILITY_6', 'PEDPEDAL_LOCATION_6', 'BAC_RESULT_6', 'CELL_PHONE_USE_6'
])

print("Dataframe size after adding dummies to peoples columns: ", cvp_ohe.shape)


# In[16]:


cvp_ohe.head(5)

