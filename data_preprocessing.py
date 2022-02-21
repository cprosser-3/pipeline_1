import sys
import time

import numpy as np
import pyspark.sql.functions as F
import regex as re

from IPython.display import display
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from typing import *


T = TypeVar('T')


class DataField():
    '''Class for handling data field functions'''
    __slots__ = [
          'FieldType'
        , 'CleanFunction'
        , 'FunctionType'
    ]
    
    def __init__(self, FieldType: T, CleanFunction: F.UserDefinedFunction, FunctionType: T = None):
        self.FieldType = FieldType
        self.CleanFunction = CleanFunction
        self.FunctionType = FunctionType if FunctionType else FieldType


def handle_pattern(pattern: str, val: str) -> str:
    '''Handles regex pattern match'''
    return val if re.match(pattern, val) else None
    
    
def clean_date(val: str) -> str:
    '''Cleans: datetime, no validation''' # TODO: fix
    try:
        return val
    except:
        return None


def clean_gametype(val: str) -> str:
    '''Cleans: GameType'''
    try:
        val = val.lower()
        return val if val in 'regular,playoff' else None
    except:
        return None


def clean_int(val: int) -> int:
    '''Cleans: generic integer, no validation'''
    try:
        return val
    except:
        return None


def clean_player(val: str) -> str:
    '''Cleans: player fields'''
    try:
        return handle_pattern(r'[A-Z][.] [A-Z][a-z]+', val)
    except:
        return None


def clean_player_team(val: str) -> str:
    '''Cleans: player - team fields'''
    try:
        if val == 'Team':
            return 'team'
        return re.match(r'.[.] .+ [-] ([a-z]+)[a-z]{2}\d{2}', val).groups()[0]
    except:
        return None


def clean_reboundtype(val: str) -> int:
    '''Cleans: ReboundType, converts to bit'''
    try:
        return val.lower() == 'offensive'
    except:
        return None


def clean_shottype(val: str) -> str:
    '''Cleans: JumpType'''
    try:
        return handle_pattern(r'[23][-]pt (?:[a-z]+\s?)+', val)
    except:
        return None


def clean_shot_outcome(val: str) -> int:
    '''Cleans: shot attempts, converts to bit'''
    try:
        return val.lower() == 'make'
    except:
        return None


def clean_str(val: str) -> str:
    '''Cleans: generic string, no validation'''
    try:
        return val
    except:
        return None


def clean_team(val: str) -> str:
    '''Cleans: team fields'''
    try:
        return handle_pattern(r'[A-Z]{3}', str(val))
    except:
        return None
    
    
DATA_FIELDS = {
      'Url': DataField(StringType, clean_str)
    , 'GameType': DataField(StringType, clean_gametype)
    , 'Location': DataField(StringType, clean_str)
    , 'Date': DataField(StringType, clean_date) # np.datetime64?
    , 'Time': DataField(StringType, clean_date) # np.datetime64?
    , 'WinningTeam': DataField(StringType, clean_team)
    , 'Quarter': DataField(IntegerType, clean_int)
    , 'SecLeft': DataField(IntegerType, clean_int)
    , 'AwayTeam': DataField(StringType, clean_team)
    , 'AwayPlay': DataField(StringType, clean_str)
    , 'AwayScore': DataField(IntegerType, clean_int)
    , 'HomeTeam': DataField(StringType, clean_team)
    , 'HomePlay': DataField(StringType, clean_str)
    , 'HomeScore': DataField(IntegerType, clean_int)
    , 'Shooter': DataField(StringType, clean_player_team)
    , 'ShotType': DataField(StringType, clean_shottype)
    , 'ShotOutcome': DataField(StringType, clean_shot_outcome, IntegerType)
    , 'ShotDist': DataField(IntegerType, clean_int)
    , 'Assister': DataField(StringType, clean_player_team)
    , 'Blocker': DataField(StringType, clean_player_team)
    , 'FoulType': DataField(StringType, clean_str) # category conversion?
    , 'Fouler': DataField(StringType, clean_player_team)
    , 'Fouled': DataField(StringType, clean_player_team)
    , 'Rebounder': DataField(StringType, clean_player_team)
    , 'ReboundType': DataField(StringType, clean_reboundtype, IntegerType)
    , 'ViolationPlayer': DataField(StringType, clean_player_team)
    , 'ViolationType': DataField(StringType, clean_str) # category conversion?
    , 'TimeoutTeam': DataField(StringType, clean_team)
    , 'FreeThrowShooter': DataField(StringType, clean_player_team)
    , 'FreeThrowOutcome': DataField(StringType, clean_shot_outcome, IntegerType)
    , 'FreeThrowNum': DataField(StringType, clean_str) # category conversion?
    , 'EnterGame': DataField(StringType, clean_player_team)
    , 'LeaveGame': DataField(StringType, clean_player_team)
    , 'TurnoverPlayer': DataField(StringType, clean_player_team)
    , 'TurnoverType': DataField(StringType, clean_str) # category conversion?
    , 'TurnoverCause': DataField(StringType, clean_str) # ?
    , 'TurnoverCauser': DataField(StringType, clean_player_team)
    , 'JumpballAwayPlayer': DataField(StringType, clean_player_team)
    , 'JumpballHomePlayer': DataField(StringType, clean_player_team)
    , 'JumpballPoss': DataField(StringType, clean_player_team)
}

    
def main():
    '''
    arg1: Get path
        ex: /project/ds5559/group2nba/NBA-PBP_201[0-9]-{1[0-9],20}.csv
    
    arg2: Post path
        ex: /project/ds5559/group2nba/data.csv
    '''
    spark = SparkSession \
        .builder \
        .appName('group2nba') \
        .getOrCreate()
    
    schema = StructType([StructField(k, v.FieldType()) for k, v in DATA_FIELDS.items()])
    
    print('Retrieving raw data...')
    try:
        df = spark.read \
            .format('csv') \
            .option('header', True) \
            .schema(schema) \
            .load(sys.argv[1])
    except:
        print('Error retrieving data from ' + sys.argv[1])
        return
    
    print(f'Raw dataframe count: {df.count()}')
    print(df.printSchema())
    time.wait(1)
    print(df.head(2))
    time.wait(1)
    print('Cleaning data...')
    
    for k, v in DATA_FIELDS.items():
        print(f'Cleaning {k}...')
        df = df.withColumn(k, F.UserDefinedFunction(v.CleanFunction, v.FunctionType())(k))
    
    print('Cleaning complete...')
    display(df.printSchema())
    time.wait(1)
    display(df.head(2))
    time.wait(1)
    
    print('Saving clean data...')
    try:
        df.write.csv(sys.argv[2])
    except:
        print('Error saving data to ' + sys.argv[2])
        return
    
    print('Clean data saved...')
    
        
if __name__ == '__main__':
    main()