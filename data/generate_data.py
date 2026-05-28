import numpy as np
import pandas as pd
import random
import os
from datetime import timedelta

np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MARATHONS = ['Tokyo', 'Boston', 'London', 'Berlin', 'Chicago', 'New York City']
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

MONTHS = {
    'Tokyo': 3,
    'Boston': 4,
    'London': 4,
    'Berlin': 9,
    'Chicago': 10,
    'New York City': 11
}

CITIES = {
    'Tokyo': {'city': 'Tokyo', 'country': 'Japan', 'elevation_gain_m': 50, 'course_type': 'flat'},
    'Boston': {'city': 'Boston', 'country': 'USA', 'elevation_gain_m': 230, 'course_type': 'net_downhill'},
    'London': {'city': 'London', 'country': 'UK', 'elevation_gain_m': 30, 'course_type': 'flat'},
    'Berlin': {'city': 'Berlin', 'country': 'Germany', 'elevation_gain_m': 25, 'course_type': 'flat_fast'},
    'Chicago': {'city': 'Chicago', 'country': 'USA', 'elevation_gain_m': 35, 'course_type': 'flat'},
    'New York City': {'city': 'New York City', 'country': 'USA', 'elevation_gain_m': 250, 'course_type': 'hilly'},
}

WINNERS_DATA = {
    (2018, 'Tokyo', 'M'): {'name': 'Dickson Chumba', 'country': 'KEN', 'time': '2:06:33'},
    (2018, 'Tokyo', 'F'): {'name': 'Birhane Dibaba', 'country': 'ETH', 'time': '2:23:47'},
    (2018, 'Boston', 'M'): {'name': 'Yuki Kawauchi', 'country': 'JPN', 'time': '2:15:58'},
    (2018, 'Boston', 'F'): {'name': 'Desiree Linden', 'country': 'USA', 'time': '2:39:54'},
    (2018, 'London', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:04:17'},
    (2018, 'London', 'F'): {'name': 'Vivian Cheruiyot', 'country': 'KEN', 'time': '2:18:31'},
    (2018, 'Berlin', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:01:39'},
    (2018, 'Berlin', 'F'): {'name': 'Gladys Cherono', 'country': 'KEN', 'time': '2:18:11'},
    (2018, 'Chicago', 'M'): {'name': 'Mo Farah', 'country': 'GBR', 'time': '2:05:11'},
    (2018, 'Chicago', 'F'): {'name': 'Brigid Kosgei', 'country': 'KEN', 'time': '2:18:35'},
    (2018, 'New York City', 'M'): {'name': 'Lelisa Desisa', 'country': 'ETH', 'time': '2:16:09'},
    (2018, 'New York City', 'F'): {'name': 'Mary Keitany', 'country': 'KEN', 'time': '2:22:48'},

    (2019, 'Tokyo', 'M'): {'name': 'Birhanu Legese', 'country': 'ETH', 'time': '2:04:48'},
    (2019, 'Tokyo', 'F'): {'name': 'Ruti Aga', 'country': 'ETH', 'time': '2:23:29'},
    (2019, 'Boston', 'M'): {'name': 'Lawrence Cherono', 'country': 'KEN', 'time': '2:07:57'},
    (2019, 'Boston', 'F'): {'name': 'Worknesh Degefa', 'country': 'ETH', 'time': '2:23:31'},
    (2019, 'London', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:02:37'},
    (2019, 'London', 'F'): {'name': 'Brigid Kosgei', 'country': 'KEN', 'time': '2:18:20'},
    (2019, 'Berlin', 'M'): {'name': 'Kenenisa Bekele', 'country': 'ETH', 'time': '2:01:41'},
    (2019, 'Berlin', 'F'): {'name': 'Ashete Bekere', 'country': 'ETH', 'time': '2:20:14'},
    (2019, 'Chicago', 'M'): {'name': 'Lawrence Cherono', 'country': 'KEN', 'time': '2:05:09'},
    (2019, 'Chicago', 'F'): {'name': 'Brigid Kosgei', 'country': 'KEN', 'time': '2:14:04'},
    (2019, 'New York City', 'M'): {'name': 'Geoffrey Kamworor', 'country': 'KEN', 'time': '2:08:13'},
    (2019, 'New York City', 'F'): {'name': 'Joyciline Jepkosgei', 'country': 'KEN', 'time': '2:22:38'},

    (2020, 'Tokyo', 'M'): {'name': 'Birhanu Legese', 'country': 'ETH', 'time': '2:04:15'},
    (2020, 'Tokyo', 'F'): {'name': 'Lonah Salpeter', 'country': 'ISR', 'time': '2:17:45'},
    (2020, 'London', 'M'): {'name': 'Shura Kitata', 'country': 'ETH', 'time': '2:05:41'},
    (2020, 'London', 'F'): {'name': 'Brigid Kosgei', 'country': 'KEN', 'time': '2:18:58'},

    (2021, 'Boston', 'M'): {'name': 'Benson Kipruto', 'country': 'KEN', 'time': '2:09:51'},
    (2021, 'Boston', 'F'): {'name': 'Edna Kiplagat', 'country': 'KEN', 'time': '2:24:59'},
    (2021, 'London', 'M'): {'name': 'Sisay Lemma', 'country': 'ETH', 'time': '2:04:01'},
    (2021, 'London', 'F'): {'name': 'Joyciline Jepkosgei', 'country': 'KEN', 'time': '2:17:43'},
    (2021, 'Berlin', 'M'): {'name': 'Guye Adola', 'country': 'ETH', 'time': '2:05:45'},
    (2021, 'Berlin', 'F'): {'name': 'Gotytom Gebreslase', 'country': 'ETH', 'time': '2:20:09'},
    (2021, 'Chicago', 'M'): {'name': 'Seifu Tura', 'country': 'ETH', 'time': '2:06:12'},
    (2021, 'Chicago', 'F'): {'name': 'Ruth Chepngetich', 'country': 'KEN', 'time': '2:22:31'},
    (2021, 'New York City', 'M'): {'name': 'Albert Korir', 'country': 'KEN', 'time': '2:08:36'},
    (2021, 'New York City', 'F'): {'name': 'Peres Jepchirchir', 'country': 'KEN', 'time': '2:22:39'},

    (2022, 'Tokyo', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:02:40'},
    (2022, 'Tokyo', 'F'): {'name': 'Brigid Kosgei', 'country': 'KEN', 'time': '2:18:18'},
    (2022, 'Boston', 'M'): {'name': 'Evans Chebet', 'country': 'KEN', 'time': '2:06:51'},
    (2022, 'Boston', 'F'): {'name': 'Peres Jepchirchir', 'country': 'KEN', 'time': '2:21:01'},
    (2022, 'London', 'M'): {'name': 'Amos Kipruto', 'country': 'KEN', 'time': '2:04:02'},
    (2022, 'London', 'F'): {'name': 'Yalemzerf Yehualaw', 'country': 'ETH', 'time': '2:17:25'},
    (2022, 'Berlin', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:01:09'},
    (2022, 'Berlin', 'F'): {'name': 'Tigst Assefa', 'country': 'ETH', 'time': '2:18:11'},
    (2022, 'Chicago', 'M'): {'name': 'Benson Kipruto', 'country': 'KEN', 'time': '2:05:58'},
    (2022, 'Chicago', 'F'): {'name': 'Ruth Chepngetich', 'country': 'KEN', 'time': '2:14:18'},
    (2022, 'New York City', 'M'): {'name': 'Evans Chebet', 'country': 'KEN', 'time': '2:07:09'},
    (2022, 'New York City', 'F'): {'name': 'Sharon Lokedi', 'country': 'KEN', 'time': '2:23:23'},

    (2023, 'Tokyo', 'M'): {'name': 'Deso Gelmisa', 'country': 'ETH', 'time': '2:05:22'},
    (2023, 'Tokyo', 'F'): {'name': 'Rosemary Wanjiru', 'country': 'KEN', 'time': '2:16:00'},
    (2023, 'Boston', 'M'): {'name': 'Evans Chebet', 'country': 'KEN', 'time': '2:07:37'},
    (2023, 'Boston', 'F'): {'name': 'Hellen Obiri', 'country': 'KEN', 'time': '2:21:38'},
    (2023, 'London', 'M'): {'name': 'Kelvin Kiptum', 'country': 'KEN', 'time': '2:04:01'},
    (2023, 'London', 'F'): {'name': 'Sifan Hassan', 'country': 'NED', 'time': '2:18:33'},
    (2023, 'Berlin', 'M'): {'name': 'Eliud Kipchoge', 'country': 'KEN', 'time': '2:02:42'},
    (2023, 'Berlin', 'F'): {'name': 'Tigst Assefa', 'country': 'ETH', 'time': '2:11:53'},
    (2023, 'Chicago', 'M'): {'name': 'Kelvin Kiptum', 'country': 'KEN', 'time': '2:00:35'},
    (2023, 'Chicago', 'F'): {'name': 'Sifan Hassan', 'country': 'NED', 'time': '2:13:44'},
    (2023, 'New York City', 'M'): {'name': 'Tamirat Tola', 'country': 'ETH', 'time': '2:05:31'},
    (2023, 'New York City', 'F'): {'name': 'Hellen Obiri', 'country': 'KEN', 'time': '2:20:36'},

    (2024, 'Tokyo', 'M'): {'name': 'Benson Kipruto', 'country': 'KEN', 'time': '2:03:41'},
    (2024, 'Tokyo', 'F'): {'name': 'Sutume Kebede', 'country': 'ETH', 'time': '2:16:31'},
    (2024, 'Boston', 'M'): {'name': 'Sisay Lemma', 'country': 'ETH', 'time': '2:06:17'},
    (2024, 'Boston', 'F'): {'name': 'Hellen Obiri', 'country': 'KEN', 'time': '2:22:20'},
    (2024, 'London', 'M'): {'name': 'Alexander Mutiso Munyao', 'country': 'KEN', 'time': '2:04:01'},
    (2024, 'London', 'F'): {'name': 'Peres Jepchirchir', 'country': 'KEN', 'time': '2:16:16'},
    (2024, 'Berlin', 'M'): {'name': 'Milkesa Mengesha', 'country': 'ETH', 'time': '2:04:21'},
    (2024, 'Berlin', 'F'): {'name': 'Tigist Ketema', 'country': 'ETH', 'time': '2:20:01'},
    (2024, 'Chicago', 'M'): {'name': 'John Korir', 'country': 'KEN', 'time': '2:02:44'},
    (2024, 'Chicago', 'F'): {'name': 'Ruth Chepngetich', 'country': 'KEN', 'time': '2:09:56'},
    (2024, 'New York City', 'M'): {'name': 'Abdi Nageeye', 'country': 'NED', 'time': '2:07:39'},
    (2024, 'New York City', 'F'): {'name': 'Sheila Chepkirui', 'country': 'KEN', 'time': '2:24:35'},

    (2025, 'Tokyo', 'M'): {'name': 'Tadese Takele', 'country': 'ETH', 'time': '2:03:23'},
    (2025, 'Tokyo', 'F'): {'name': 'Sutume Kebede', 'country': 'ETH', 'time': '2:16:00'},
    (2025, 'Boston', 'M'): {'name': 'John Korir', 'country': 'KEN', 'time': '2:04:45'},
    (2025, 'Boston', 'F'): {'name': 'Sharon Lokedi', 'country': 'KEN', 'time': '2:17:22'},
    (2025, 'London', 'M'): {'name': 'Sabastian Sawe', 'country': 'KEN', 'time': '1:59:30'},
    (2025, 'London', 'F'): {'name': 'Tigst Assefa', 'country': 'ETH', 'time': '2:15:50'},
    (2025, 'Berlin', 'M'): {'name': 'Hailemaryam Kiros', 'country': 'ETH', 'time': '2:04:11'},
    (2025, 'Berlin', 'F'): {'name': 'Rosemary Wanjiru', 'country': 'KEN', 'time': '2:19:34'},
    (2025, 'Chicago', 'M'): {'name': 'Sabastian Sawe', 'country': 'KEN', 'time': '2:02:04'},
    (2025, 'Chicago', 'F'): {'name': 'Hawi Feysa', 'country': 'ETH', 'time': '2:17:25'},
    (2025, 'New York City', 'M'): {'name': 'Jacob Kiplimo', 'country': 'UGA', 'time': '2:01:07'},
    (2025, 'New York City', 'F'): {'name': 'Hellen Obiri', 'country': 'KEN', 'time': '2:18:35'},
}

CANCELLED_RACES = [
    (2020, 'Boston'),
    (2020, 'Berlin'),
    (2020, 'Chicago'),
    (2020, 'New York City'),
    (2021, 'Tokyo'),
]

COURSE_RECORD_HOLDERS = {
    'Tokyo': {'M': {'name': 'Eliud Kipchoge', 'time': '2:02:40', 'year': 2022},
              'F': {'name': 'Sutume Kebede', 'time': '2:16:00', 'year': 2025}},
    'Boston': {'M': {'name': 'Geoffrey Mutai', 'time': '2:03:02', 'year': 2011},
               'F': {'name': 'Buzunesh Deba', 'time': '2:19:59', 'year': 2014}},
    'London': {'M': {'name': 'Kelvin Kiptum', 'time': '2:01:25', 'year': 2023},
               'F': {'name': 'Paula Radcliffe', 'time': '2:15:25', 'year': 2003}},
    'Berlin': {'M': {'name': 'Kelvin Kiptum', 'time': '2:01:53', 'year': 2023},
               'F': {'name': 'Tigst Assefa', 'time': '2:11:53', 'year': 2023}},
    'Chicago': {'M': {'name': 'Kelvin Kiptum', 'time': '2:00:35', 'year': 2023},
                'F': {'name': 'Ruth Chepngetich', 'time': '2:09:56', 'year': 2024}},
    'New York City': {'M': {'name': 'Geoffrey Mutai', 'time': '2:05:06', 'year': 2011},
                      'F': {'name': 'Margaret Okayo', 'time': '2:22:31', 'year': 2003}},
}

COUNTRY_WEIGHTS = {
    'KEN': 0.08,
    'ETH': 0.08,
    'USA': 0.20,
    'JPN': 0.07,
    'GBR': 0.06,
    'DEU': 0.05,
    'BRA': 0.03,
    'FRA': 0.04,
    'ITA': 0.03,
    'CHN': 0.03,
    'CAN': 0.03,
    'AUS': 0.03,
    'NED': 0.02,
    'ESP': 0.02,
}

OTHER_COUNTRIES = [
    'SWE', 'NOR', 'DEN', 'FIN', 'POL', 'CZE', 'AUT', 'SUI', 'BEL',
    'POR', 'IRL', 'RUS', 'UKR', 'ROU', 'HUN', 'TUR', 'ISR', 'RSA',
    'MAR', 'EGY', 'NGR', 'ETH', 'UGA', 'TAN', 'MEX', 'COL', 'ARG',
    'CHL', 'PER', 'ECU', 'VEN', 'KOR', 'IND', 'THA', 'PHL', 'MYS',
    'IDN', 'NZL', 'SGP', 'HKG', 'TWN', 'BRN', 'QAT', 'UAE', 'KSA',
]

GENDER_DIST = {'M': 0.65, 'F': 0.35}

FIRST_NAMES_M = {
    'KEN': ['Eliud', 'Kipchoge', 'Geoffrey', 'Lawrence', 'Evans', 'Benson', 'Brimin', 'Daniel',
            'Vincent', 'Albert', 'Amos', 'John', 'Samuel', 'Wesley', 'Micah', 'Ferguson',
            'Alexander', 'Sabastian', 'Hillary', 'Jonathan', 'Paul', 'Patrick', 'Benjamin',
            'James', 'Michael', 'David', 'Peter', 'Joseph', 'Thomas', 'Richard'],
    'ETH': ['Kenenisa', 'Birhanu', 'Sisay', 'Guye', 'Deso', 'Tamirat', 'Seifu', 'Lelisa',
            'Shura', 'Tadese', 'Milkesa', 'Hailemaryam', 'Hawi', 'Yemane', 'Tsegaye',
            'Abebe', 'Mule', 'Getaneh', 'Mosinet', 'Bazezew', 'Deriba', 'Tadesse',
            'Worknesh', 'Kibrom', 'Yohannes', 'Girmawit', 'Belete', 'Asnakech', 'Dinknesh'],
    'USA': ['John', 'Michael', 'David', 'James', 'Robert', 'William', 'Richard', 'Joseph',
            'Thomas', 'Christopher', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark',
            'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian',
            'George', 'Timothy', 'Ronald', 'Edward', 'Jason', 'Jeffrey', 'Ryan'],
    'JPN': ['Yuki', 'Satoshi', 'Hiroshi', 'Takeshi', 'Kenji', 'Yusuke', 'Shohei', 'Ryu',
            'Kenta', 'Daichi', 'Takumi', 'Haruki', 'Shota', 'Yuto', 'Koki', 'Ren',
            'Hiroto', 'Sota', 'Ryota', 'Yuma', 'Kaito', 'Haruto', 'Sora', 'Yusei'],
    'GBR': ['James', 'Oliver', 'Harry', 'Jack', 'Charlie', 'Freddie', 'George', 'William',
            'Thomas', 'Henry', 'Arthur', 'Leo', 'Oscar', 'Alfie', 'Theodore', 'Edward',
            'Alexander', 'Max', 'Lucas', 'Daniel', 'Mohammed', 'Noah', 'Ethan', 'Samuel'],
    'DEU': ['Lukas', 'Maximilian', 'Leon', 'Paul', 'Tim', 'Felix', 'David', 'Jan',
            'Lukas', 'Jonas', 'Alexander', 'Marcel', 'Stefan', 'Markus', 'Thomas',
            'Michael', 'Christian', 'Daniel', 'Martin', 'Andreas', 'Florian', 'Tobias'],
    'BRA': ['Lucas', 'Gabriel', 'Pedro', 'Rafael', 'Joao', 'Guilherme', 'Bruno', 'Marcos',
            'Felipe', 'Andre', 'Thiago', 'Carlos', 'Rodrigo', 'Mateus', 'Diego', 'Daniel',
            'Leandro', 'Gustavo', 'Ricardo', 'Fernando', 'Alexandre', 'Eduardo', 'Victor'],
    'FRA': ['Louis', 'Jules', 'Gabriel', 'Raphal', 'Arthur', 'Hugo', 'Leo', 'Paul',
            'Lucas', 'Nathan', 'Theo', 'Mathis', 'Enzo', 'Maxime', 'Antoine', 'Alexandre',
            'Baptiste', 'Pierre', 'Nicolas', 'Julien', 'Thomas', 'Alexandre', 'Marc'],
    'ITA': ['Marco', 'Alessandro', 'Luca', 'Lorenzo', 'Andrea', 'Matteo', 'Francesco', 'Davide',
            'Giuseppe', 'Roberto', 'Stefano', 'Federico', 'Giovanni', 'Antonio', 'Paolo',
            'Simone', 'Fabio', 'Alberto', 'Giorgio', 'Filippo', 'Riccardo', 'Tommaso'],
    'CHN': ['Wei', 'Lei', 'Jie', 'Ming', 'Jun', 'Peng', 'Chen', 'Hao', 'Tao', 'Liang',
            'Xin', 'Jian', 'Guang', 'Dong', 'Sheng', 'Yong', 'Zhi', 'Feng', 'Hui', 'Bo'],
    'CAN': ['Liam', 'Noah', 'William', 'Oliver', 'Benjamin', 'Lucas', 'Ethan', 'Jack',
            'Owen', 'Alexander', 'James', 'Daniel', 'Henry', 'Matthew', 'Ryan', 'Nathan',
            'Dylan', 'Carter', 'Connor', 'Brandon', 'Kevin', 'Justin', 'Tyler'],
    'AUS': ['Oliver', 'William', 'Jack', 'Noah', 'Thomas', 'James', 'Henry', 'Liam',
            'Ethan', 'Lucas', 'Alexander', 'Daniel', 'Benjamin', 'Samuel', 'Matthew',
            'Ryan', 'Nathan', 'Dylan', 'Carter', 'Lachlan', 'Hamish', 'Angus', 'Callum'],
    'NED': ['Daan', 'Sem', 'Lucas', 'Milan', 'Levi', 'Finn', 'Luuk', 'Tim', 'Jesse',
            'Nolan', 'Thijs', 'Lars', 'Bram', 'Max', 'Mats', 'Teun', 'Sven', 'Joris',
            'Pim', 'Wouter', 'Pieter', 'Jan', 'Hendrik'],
    'ESP': ['Alejandro', 'Daniel', 'Pablo', 'Hugo', 'Martin', 'Lucas', 'Adrian', 'Mateo',
            'Leo', 'Alvaro', 'David', 'Mario', 'Sergio', 'Javier', 'Marcos', 'Diego',
            'Carlos', 'Jorge', 'Miguel', 'Rafael', 'Pedro', 'Fernando', 'Antonio'],
}

FIRST_NAMES_F = {
    'KEN': ['Brigid', 'Mary', 'Joyciline', 'Peres', 'Hellen', 'Ruth', 'Sharon', 'Vivian',
            'Gladys', 'Ruti', 'Rosemary', 'Sheila', 'Margaret', 'Agnes', 'Faith',
            'Linet', 'Hawi', 'Edna', 'Sally', 'Caroline', 'Emily', 'Lucy', 'Esther',
            'Jackline', 'Mercy', 'Irene', 'Beatrice', 'Winfred', 'Nancy', 'Diana'],
    'ETH': ['Birhane', 'Worknesh', 'Ashete', 'Gotytom', 'Tigst', 'Yalemzerf', 'Sutume',
            'Tigist', 'Hawi', 'Lonah', 'Ruti', 'Abeba', 'Koren', 'Meselech', 'Abera',
            'Shitaye', 'Alemu', 'Bekelech', 'Hiwot', 'Kidist', 'Tigist', 'Mekonnen',
            'Genet', 'Aberash', 'Marta', 'Aster', 'Frehiwot', 'Buzunesh', 'Alem'],
    'USA': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara', 'Elizabeth', 'Susan',
            'Jessica', 'Sarah', 'Karen', 'Lisa', 'Nancy', 'Betty', 'Margaret', 'Sandra',
            'Ashley', 'Dorothy', 'Kimberly', 'Emily', 'Donna', 'Michelle', 'Carol',
            'Amanda', 'Melissa', 'Deborah', 'Stephanie', 'Rebecca', 'Sharon', 'Laura'],
    'JPN': ['Yui', 'Sakura', 'Haruka', 'Mio', 'Rina', 'Aoi', 'Yuna', 'Hina', ' Mei',
            'Rio', 'Miyu', 'Hana', 'Akari', 'Yuka', 'Nanami', 'Saki', 'Moe', 'Chloe',
            'Ayumi', 'Kaede', 'Maki', 'Nao', 'Shiori', 'Tomomi', 'Yoko', 'Midori'],
    'GBR': ['Olivia', 'Amelia', 'Isla', 'Emily', 'Charlotte', 'Sophie', 'Grace', 'Alice',
            'Ella', 'Florence', 'Evie', 'Phoebe', 'Poppy', 'Jessica', 'Lily', 'Mia',
            'Sophia', 'Isabella', 'Eva', 'Molly', 'Lucy', 'Eleanor', 'Chloe', 'Freya'],
    'DEU': ['Hannah', 'Mia', 'Emma', 'Sophia', 'Lina', 'Anna', 'Marie', 'Leni', 'Clara',
            'Lena', 'Leonie', 'Julia', 'Lara', 'Laura', 'Sarah', 'Katharina', 'Anne',
            'Sophie', 'Johanna', 'Amelie', 'Finja', 'Pia', 'Maja', 'Luisa'],
    'BRA': ['Ana', 'Maria', 'Julia', 'Camila', 'Fernanda', 'Patricia', 'Amanda', 'Bruna',
            'Carolina', 'Gabriela', 'Larissa', 'Luana', 'Mariana', 'Natalia', 'Isabela',
            'Rafaela', 'Jessica', 'Letcia', 'Daniela', 'Beatriz', 'Alessandra', 'Paula'],
    'FRA': ['Emma', 'Jade', 'Louise', 'Alice', 'Chlo', 'Lina', 'Lea', 'Manon', 'Rose',
            'Anna', 'Camille', 'Juliette', 'Jeanne', 'Pauline', 'Marie', 'Charlotte',
            'Clara', 'Lucie', 'Sarah', 'Lou', 'Nina', 'Eva', 'Ines', 'Julia'],
    'ITA': ['Sofia', 'Giulia', 'Aurora', 'Beatrice', 'Alice', 'Ginevra', 'Emma', 'Giorgia',
            'Matilde', 'Chiara', 'Anna', 'Vittoria', 'Azzurra', 'Camilla', 'Martina',
            'Alessia', 'Sara', 'Elena', 'Valentina', 'Francesca', 'Roberta', 'Federica'],
    'CHN': ['Wei', 'Fang', 'Jie', 'Ming', 'Xiu', 'Lan', 'Yan', 'Mei', 'Ling', 'Hua',
            'Ying', 'Li', 'Xin', 'Hong', 'Yong', 'Ping', 'Shuang', 'Jing', 'Chun', 'Na'],
    'CAN': ['Olivia', 'Emma', 'Charlotte', 'Amelia', 'Sophia', 'Ava', 'Isabella', 'Mia',
            'Evelyn', 'Harper', 'Lily', 'Chloe', 'Grace', 'Zoey', 'Emily', 'Sarah',
            'Abigail', 'Ella', 'Scarlett', 'Victoria', 'Sophie', 'Nora', 'Hannah'],
    'AUS': ['Charlotte', 'Olivia', 'Amelia', 'Mia', 'Harper', 'Sophia', 'Chloe', 'Grace',
            'Emily', 'Ella', 'Isla', 'Matilda', 'Alice', 'Lily', 'Abigail', 'Zoe',
            'Sophie', 'Scarlett', 'Evie', 'Ruby', 'Ava', 'Ivy', 'Mila', 'Eva'],
    'NED': ['Emma', 'Sophie', 'Julia', 'Mila', 'Lot', 'Olivia', 'Sara', 'Saar', 'Femke',
            'Lieke', 'Anna', 'Lotte', 'Eva', 'Liv', 'Elin', 'Fien', 'Nina', 'Isa',
            'Maaike', 'Anne', 'Sanne', 'Lisanne', 'Petra', 'Marlies'],
    'ESP': ['Lucia', 'Sofia', 'Maria', 'Martina', 'Emma', 'Paula', 'Daniela', 'Valeria',
            'Alba', 'Julia', 'Carmen', 'Noa', 'Lola', 'Chloe', 'Adriana', 'Blanca',
            'Clara', 'Elena', 'Isabel', 'Laura', 'Ana', 'Rosa', 'Pilar', 'Teresa'],
}

LAST_NAMES = {
    'KEN': ['Kipchoge', 'Kipruto', 'Cherono', 'Chebet', 'Korir', 'Kamworor', 'Kosgei', 'Keitany',
            'Jepchirchir', 'Jepkosgei', 'Kiptum', 'Kipyegon', 'Chemutai', 'Kipkirui', 'Kurgat',
            'Keter', 'Bett', 'Koech', 'Rono', 'Kipkemboi', 'Sawe', 'Lokedi', 'Chepkirui',
            'Wanjiru', 'Kiptoo', 'Kiprop', 'Langat', 'Cheruiyot', 'Kimutai', 'Maiyo'],
    'ETH': ['Bekele', 'Legese', 'Lemma', 'Adola', 'Gelmisa', 'Tola', 'Tura', 'Desisa',
            'Kitata', 'Takele', 'Mengesha', 'Kiros', 'Feysa', 'Dibaba', 'Cherono', 'Assefa',
            'Yehualaw', 'Kebede', 'Gebreslase', 'Bekere', 'Aga', 'Ketema', 'Salpeter', 'Gebrhiwet',
            'Abebe', 'Werknesh', 'Mule', 'Gidey', 'Berhanu', 'Hailu'],
    'USA': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
            'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
            'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson'],
    'JPN': ['Kawauchi', 'Suzuki', 'Takahashi', 'Tanaka', 'Watanabe', 'Ito', 'Yamamoto', 'Nakamura',
            'Kobayashi', 'Kato', 'Yoshida', 'Yamada', 'Sasaki', 'Yamashita', 'Ishii', 'Hasegawa',
            'Endo', 'Fujita', 'Maeda', 'Mori', 'Ogawa', 'Nakano', 'Shimizu', 'Kimura'],
    'GBR': ['Smith', 'Jones', 'Taylor', 'Brown', 'Williams', 'Wilson', 'Johnson', 'Davies',
            'Evans', 'Thomas', 'Roberts', 'Walker', 'Wright', 'Thompson', 'White', 'Hughes',
            'Edwards', 'Green', 'Hall', 'Wood', 'Clark', 'Baker', 'Harris', 'Mitchell'],
    'DEU': ['Mueller', 'Schmidt', 'Schneider', 'Fischer', 'Weber', 'Meyer', 'Wagner', 'Becker',
            'Schulz', 'Hoffmann', 'Koch', 'Richter', 'Wolf', 'Schaefer', 'Neumann', 'Schwarz',
            'Zimmermann', 'Braun', 'Krause', 'Hartmann', 'Werner', 'Lange', 'Kroeger'],
    'BRA': ['Silva', 'Santos', 'Oliveira', 'Souza', 'Rodrigues', 'Ferreira', 'Alves', 'Pereira',
            'Lima', 'Gomes', 'Costa', 'Ribeiro', 'Martins', 'Carvalho', 'Almeida', 'Lopes',
            'Soares', 'Vieira', 'Barbosa', 'Rocha', 'Dias', 'Nascimento', 'Andrade', 'Moreira'],
    'FRA': ['Martin', 'Bernard', 'Thomas', 'Petit', 'Robert', 'Richard', 'Durand', 'Dubois',
            'Moreau', 'Laurent', 'Simon', 'Michel', 'Lefebvre', 'Leroy', 'Roux', 'David',
            'Bertrand', 'Morel', 'Fournier', 'Girard', 'Bonnet', 'Dupont', 'Lambert'],
    'ITA': ['Rossi', 'Ferrari', 'Russo', 'Bianchi', 'Esposito', 'Romano', 'Colombo', 'Ricci',
            'Marino', 'Greco', 'Bruno', 'Gallo', 'Conti', 'De Luca', 'Mancini', 'Costa',
            'Giordano', 'Rizzo', 'Lombardi', 'Moretti', 'Barbieri', 'Fontana', 'Santoro'],
    'CHN': ['Wang', 'Li', 'Zhang', 'Liu', 'Chen', 'Yang', 'Huang', 'Zhao', 'Wu', 'Zhou',
            'Xu', 'Sun', 'Ma', 'Zhu', 'Hu', 'Guo', 'Lin', 'He', 'Gao', 'Luo'],
    'CAN': ['Smith', 'Brown', 'Tremblay', 'Martin', 'Roy', 'Wilson', 'MacDonald', 'Gagnon',
            'Johnson', 'Taylor', 'Campbell', 'Anderson', 'Leblanc', 'Murphy', 'Scott',
            'Stewart', 'Clark', 'Williams', 'Mitchell', 'Thompson', 'Robinson', 'Moore'],
    'AUS': ['Smith', 'Jones', 'Williams', 'Brown', 'Wilson', 'Taylor', 'Johnson', 'White',
            'Martin', 'Anderson', 'Thompson', 'Harris', 'Walker', 'Mitchell', 'Roberts',
            'Clark', 'Robinson', 'Hall', 'Young', 'Allen', 'King', 'Scott', 'Green'],
    'NED': ['De Jong', 'Jansen', 'De Vries', 'Van den Berg', 'Van Dijk', 'Bakker', 'Janssen',
            'Visser', 'Smit', 'Meijer', 'De Boer', 'Mulder', 'De Groot', 'Bos', 'Peters',
            'Hendriks', 'Van Leeuwen', 'Dekker', 'Brouwer', 'De Wit', 'Dijkstra', 'Smit'],
    'ESP': ['Garcia', 'Rodriguez', 'Gonzalez', 'Fernandez', 'Lopez', 'Martinez', 'Sanchez',
            'Perez', 'Gomez', 'Martin', 'Jimenez', 'Ruiz', 'Hernandez', 'Diaz', 'Moreno',
            'Alvarez', 'Romero', 'Munoz', 'Alonso', 'Gutierrez', 'Navarro', 'Torres'],
}

BRAZILIAN_CITIES = [
    'Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Brasilia',
    'Curitiba', 'Recife', 'Porto Alegre', 'Fortaleza', 'Belo Horizonte',
    'Manaus', 'Belem', 'Goiania', 'Campinas', 'Florianopolis', 'Vitoria',
    'Natal', 'Joao Pessoa', 'Maceio', 'Teresina', 'Campo Grande', 'Cuiaba',
    'Aracaju', 'Londrina', 'Joinville', 'Niteroi', 'Santos', 'Uberlandia',
    'Ribeirao Preto', 'Sorocaba',
]

WEATHER_NOTES = {
    'Tokyo': {3: 'Cool, occasional rain, 8-14C avg'},
    'Boston': {4: 'Variable, can be warm or rainy, 8-18C avg'},
    'London': {4: 'Mild, often overcast, 10-16C avg'},
    'Berlin': {9: 'Mild to cool, 12-18C avg'},
    'Chicago': {10: 'Variable, can be windy, 8-16C avg'},
    'New York City': {11: 'Cool, can be windy, 5-12C avg'},
}


def time_str_to_seconds(time_str):
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def seconds_to_time_str(total_seconds):
    hours = int(total_seconds) // 3600
    minutes = (int(total_seconds) % 3600) // 60
    seconds = int(total_seconds) % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def generate_age():
    base = np.random.normal(38, 10)
    age = int(np.clip(base, 18, 75))
    return age


def generate_country():
    r = random.random()
    cum = 0.0
    for country, weight in COUNTRY_WEIGHTS.items():
        cum += weight
        if r < cum:
            return country
    return random.choice(OTHER_COUNTRIES)


def generate_name(gender, country):
    if gender == 'M':
        first_pool = FIRST_NAMES_M.get(country, FIRST_NAMES_M['USA'])
    else:
        first_pool = FIRST_NAMES_F.get(country, FIRST_NAMES_F['USA'])
    last_pool = LAST_NAMES.get(country, LAST_NAMES['USA'])
    first = random.choice(first_pool)
    last = random.choice(last_pool)
    return first, last


def generate_finish_time(gender, marathon, is_elite=False):
    if is_elite:
        if gender == 'M':
            base_seconds = random.randint(7200, 8400)
        else:
            base_seconds = random.randint(8040, 9300)
        noise = random.randint(-60, 60)
        return base_seconds + noise

    category = random.choices(
        ['elite', 'advanced', 'intermediate', 'recreational'],
        weights=[0.02, 0.18, 0.40, 0.40],
        k=1
    )[0]

    if gender == 'M':
        ranges = {
            'elite': (7320, 8400),
            'advanced': (8400, 10200),
            'intermediate': (10200, 12600),
            'recreational': (12600, 18000),
        }
    else:
        ranges = {
            'elite': (8040, 9300),
            'advanced': (9300, 11400),
            'intermediate': (11400, 13800),
            'recreational': (13800, 19800),
        }

    low, high = ranges[category]

    course_penalty = {
        'flat': 0,
        'flat_fast': -random.randint(0, 120),
        'net_downhill': -random.randint(0, 60),
        'hilly': random.randint(120, 360),
    }
    course_type = CITIES[marathon]['course_type']
    penalty = course_penalty.get(course_type, 0)

    base_seconds = random.randint(low, high) + penalty
    base_seconds = max(low, base_seconds)

    age_factor = 0
    age_val = generate_age.__code__
    return base_seconds


def generate_finish_time_with_age(gender, marathon, age):
    category = random.choices(
        ['elite', 'advanced', 'intermediate', 'recreational'],
        weights=[0.02, 0.18, 0.40, 0.40],
        k=1
    )[0]

    if gender == 'M':
        ranges = {
            'elite': (7320, 8400),
            'advanced': (8400, 10200),
            'intermediate': (10200, 12600),
            'recreational': (12600, 18000),
        }
    else:
        ranges = {
            'elite': (8040, 9300),
            'advanced': (9300, 11400),
            'intermediate': (11400, 13800),
            'recreational': (13800, 19800),
        }

    low, high = ranges[category]

    course_type = CITIES[marathon]['course_type']
    course_penalties = {
        'flat': 0,
        'flat_fast': random.randint(-120, 0),
        'net_downhill': random.randint(-60, 0),
        'hilly': random.randint(120, 360),
    }
    penalty = course_penalties.get(course_type, 0)

    base_seconds = random.randint(low, high) + penalty

    if age < 25:
        age_adj = random.randint(60, 180)
    elif 25 <= age <= 35:
        age_adj = random.randint(-30, 60)
    elif 35 < age <= 45:
        age_adj = random.randint(-10, 120)
    elif 45 < age <= 55:
        age_adj = random.randint(120, 360)
    elif 55 < age <= 65:
        age_adj = random.randint(300, 720)
    else:
        age_adj = random.randint(540, 1200)

    base_seconds += age_adj
    base_seconds = max(low, base_seconds)

    return base_seconds


def generate_split_times(finish_seconds, gender):
    distances = [5, 10, 15, 20, 21.0975, 25, 30, 35, 40]
    total_dist = 42.195

    avg_pace_per_km = finish_seconds / total_dist

    splits = []
    cumulative = 0.0
    prev_dist = 0.0

    for i, dist in enumerate(distances):
        segment_dist = dist - prev_dist
        segment_num = i + 1

        if segment_num <= 4:
            pace_factor = np.random.normal(0.97, 0.03)
        elif segment_num == 5:
            pace_factor = np.random.normal(0.98, 0.02)
        elif segment_num == 6:
            pace_factor = np.random.normal(1.00, 0.03)
        elif segment_num == 7:
            pace_factor = np.random.normal(1.02, 0.04)
        elif segment_num == 8:
            pace_factor = np.random.normal(1.05, 0.05)
        else:
            pace_factor = np.random.normal(1.08, 0.06)

        pace_factor = max(0.85, min(1.25, pace_factor))
        segment_time = segment_dist * avg_pace_per_km * pace_factor
        cumulative += segment_time
        splits.append(int(cumulative))
        prev_dist = dist

    finish_adjustment = finish_seconds - splits[-1] - (total_dist - 40) * avg_pace_per_km * np.random.normal(1.10, 0.06)
    final_segment = (total_dist - 40) * avg_pace_per_km * np.random.normal(1.10, 0.06)
    actual_finish = splits[-1] + final_segment

    ratio = finish_seconds / actual_finish if actual_finish > 0 else 1.0
    splits = [int(s * ratio) for s in splits]

    split_dict = {
        'split_5k_sec': splits[0],
        'split_10k_sec': splits[1],
        'split_15k_sec': splits[2],
        'split_20k_sec': splits[3],
        'split_half_sec': splits[4],
        'split_25k_sec': splits[5],
        'split_30k_sec': splits[6],
        'split_35k_sec': splits[7],
        'split_40k_sec': splits[8],
    }

    return split_dict


def is_race_held(year, marathon):
    for cy, cm in CANCELLED_RACES:
        if cy == year and cm == marathon:
            return False
    return True


def get_race_participants(year, marathon):
    base_participants = {
        'Tokyo': 38000,
        'Boston': 30000,
        'London': 50000,
        'Berlin': 45000,
        'Chicago': 45000,
        'New York City': 53000,
    }
    base = base_participants.get(marathon, 40000)
    variation = random.randint(-2000, 2000)

    if year == 2020:
        if marathon == 'Tokyo':
            return base + variation
        elif marathon == 'London':
            return int(base * 0.8) + variation
        return 0
    elif year == 2021:
        if marathon == 'Tokyo':
            return 0
        return int(base * 0.7) + variation

    return base + variation


def generate_winners_dataset():
    rows = []
    for (year, marathon, gender), data in WINNERS_DATA.items():
        rows.append({
            'year': year,
            'marathon': marathon,
            'gender': gender,
            'winner_name': data['name'],
            'winner_country': data['country'],
            'winning_time': data['time'],
            'winning_time_sec': time_str_to_seconds(data['time']),
            'city': CITIES[marathon]['city'],
            'country_held': CITIES[marathon]['country'],
            'month': MONTHS[marathon],
        })

    df = pd.DataFrame(rows)
    return df


def generate_race_metadata():
    rows = []
    for marathon in MARATHONS:
        for year in YEARS:
            if not is_race_held(year, marathon):
                continue

            participants = get_race_participants(year, marathon)
            month = MONTHS[marathon]
            weather = WEATHER_NOTES.get(marathon, {}).get(month, 'Mild conditions')

            cr_m = COURSE_RECORD_HOLDERS[marathon]['M']
            cr_f = COURSE_RECORD_HOLDERS[marathon]['F']

            status = 'Completed'
            if year == 2020 and marathon in ['London']:
                status = 'Modified (COVID-19)'
            elif year == 2021:
                status = 'Modified (COVID-19)'

            row = {
                'marathon': marathon,
                'year': year,
                'city': CITIES[marathon]['city'],
                'country': CITIES[marathon]['country'],
                'month': month,
                'participants_estimate': participants,
                'course_type': CITIES[marathon]['course_type'],
                'elevation_gain_m': CITIES[marathon]['elevation_gain_m'],
                'weather_notes': weather,
                'status': status,
                'course_record_male_name': cr_m['name'],
                'course_record_male_time': cr_m['time'],
                'course_record_male_year': cr_m['year'],
                'course_record_female_name': cr_f['name'],
                'course_record_female_time': cr_f['time'],
                'course_record_female_year': cr_f['year'],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_marathon_results():
    all_rows = []
    bib_counter = 1

    for year in YEARS:
        for marathon in MARATHONS:
            if not is_race_held(year, marathon):
                continue

            num_participants = get_race_participants(year, marathon)
            target_runners = int(num_participants * 0.35) + random.randint(-500, 500)
            target_runners = max(1500, target_runners)

            for _ in range(target_runners):
                gender = random.choices(['M', 'F'], weights=[GENDER_DIST['M'], GENDER_DIST['F']], k=1)[0]
                country = generate_country()
                first, last = generate_name(gender, country)
                age = generate_age()

                finish_sec = generate_finish_time_with_age(gender, marathon, age)
                splits = generate_split_times(finish_sec, gender)

                dnf = random.random() < 0.03
                dns = random.random() < 0.01

                if dns:
                    finish_time_str = 'DNS'
                    finish_sec_final = None
                    status = 'DNS'
                elif dnf:
                    finish_time_str = 'DNF'
                    finish_sec_final = None
                    status = 'DNF'
                else:
                    finish_time_str = seconds_to_time_str(finish_sec)
                    finish_sec_final = finish_sec
                    status = 'Finished'

                half_split = splits['split_half_sec']
                if finish_sec_final and half_split:
                    split_type_val = finish_sec_final - half_split
                    if split_type_val > 0:
                        split_type = 'Positive'
                    elif split_type_val < -30:
                        split_type = 'Negative'
                    else:
                        split_type = 'Even'
                else:
                    split_type = None

                row = {
                    'bib_number': bib_counter,
                    'year': year,
                    'marathon': marathon,
                    'runner_name': f"{first} {last}",
                    'gender': gender,
                    'age': age,
                    'country': country,
                    'finish_time': finish_time_str,
                    'finish_time_sec': finish_sec_final if finish_sec_final else None,
                    'status': status,
                    'split_type': split_type,
                    'split_5k_sec': splits['split_5k_sec'] if status == 'Finished' else None,
                    'split_10k_sec': splits['split_10k_sec'] if status == 'Finished' else None,
                    'split_15k_sec': splits['split_15k_sec'] if status == 'Finished' else None,
                    'split_20k_sec': splits['split_20k_sec'] if status == 'Finished' else None,
                    'split_half_sec': splits['split_half_sec'] if status == 'Finished' else None,
                    'split_25k_sec': splits['split_25k_sec'] if status == 'Finished' else None,
                    'split_30k_sec': splits['split_30k_sec'] if status == 'Finished' else None,
                    'split_35k_sec': splits['split_35k_sec'] if status == 'Finished' else None,
                    'split_40k_sec': splits['split_40k_sec'] if status == 'Finished' else None,
                }
                all_rows.append(row)
                bib_counter += 1

    df = pd.DataFrame(all_rows)
    return df


def add_winner_flags(df):
    for (year, marathon, gender), data in WINNERS_DATA.items():
        mask = (
            (df['year'] == year) &
            (df['marathon'] == marathon) &
            (df['gender'] == gender) &
            (df['runner_name'] == data['name'])
        )
        if not mask.any():
            new_row = {
                'bib_number': 0,
                'year': year,
                'marathon': marathon,
                'runner_name': data['name'],
                'gender': gender,
                'age': random.randint(25, 35),
                'country': data['country'],
                'finish_time': data['time'],
                'finish_time_sec': time_str_to_seconds(data['time']),
                'status': 'Finished',
                'split_type': 'Even',
                'split_5k_sec': None,
                'split_10k_sec': None,
                'split_15k_sec': None,
                'split_20k_sec': None,
                'split_half_sec': None,
                'split_25k_sec': None,
                'split_30k_sec': None,
                'split_35k_sec': None,
                'split_40k_sec': None,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def generate_brazilian_runners(df):
    bra_df = df[(df['country'] == 'BRA') & (df['status'] == 'Finished')].copy()

    bra_df['home_city'] = [random.choice(BRAZILIAN_CITIES) for _ in range(len(bra_df))]
    bra_df['training_years'] = bra_df['age'].apply(lambda a: max(1, min(a - 18 + random.randint(-3, 5), 40)))
    bra_df['previous_marathons'] = bra_df['training_years'].apply(lambda t: max(0, min(int(np.random.exponential(3)), t * 2)))

    bra_df['personal_best_sec'] = bra_df['finish_time_sec'].apply(
        lambda x: max(7200, x - random.randint(0, 600)) if pd.notna(x) else None
    )
    bra_df['personal_best'] = bra_df['personal_best_sec'].apply(
        lambda x: seconds_to_time_str(int(x)) if pd.notna(x) else None
    )

    bra_df['boston_qualified'] = bra_df.apply(
        lambda row: 1 if (row['gender'] == 'M' and pd.notna(row['finish_time_sec']) and row['finish_time_sec'] <= 10800) or
                     (row['gender'] == 'F' and pd.notna(row['finish_time_sec']) and row['finish_time_sec'] <= 12600) else 0,
        axis=1
    )

    def pace_per_km(sec):
        if pd.isna(sec):
            return None
        return round(sec / 42.195, 2)

    bra_df['avg_pace_sec_per_km'] = bra_df['finish_time_sec'].apply(pace_per_km)

    return bra_df


def generate_pace_splits_analysis(df):
    finished = df[df['status'] == 'Finished'].copy()

    segments = [
        ('5k', 5, 'split_5k_sec', None),
        ('10k', 5, 'split_10k_sec', 'split_5k_sec'),
        ('15k', 5, 'split_15k_sec', 'split_10k_sec'),
        ('20k', 5, 'split_20k_sec', 'split_15k_sec'),
        ('half', 1.0975, 'split_half_sec', 'split_20k_sec'),
        ('25k', 5, 'split_25k_sec', 'split_half_sec'),
        ('30k', 5, 'split_30k_sec', 'split_25k_sec'),
        ('35k', 5, 'split_35k_sec', 'split_30k_sec'),
        ('40k', 5, 'split_40k_sec', 'split_35k_sec'),
    ]

    rows = []
    for _, row in finished.iterrows():
        for seg_name, seg_dist, split_col, prev_col in segments:
            split_sec = row[split_col]
            if pd.isna(split_sec):
                continue

            if prev_col and pd.notna(row.get(prev_col)):
                prev_sec = row[prev_col]
                segment_time = split_sec - prev_sec
            else:
                segment_time = split_sec

            if segment_time <= 0:
                continue

            pace_per_km = round(segment_time / seg_dist, 2)

            if seg_name == '5k':
                segment_type = 'Start'
            elif seg_name in ['10k', '15k', '20k']:
                segment_type = 'Early Middle'
            elif seg_name in ['half', '25k']:
                segment_type = 'Middle'
            elif seg_name == '30k':
                segment_type = 'Late Middle'
            elif seg_name == '35k':
                segment_type = 'Late'
            else:
                segment_type = 'Final'

            rows.append({
                'bib_number': row['bib_number'],
                'year': row['year'],
                'marathon': row['marathon'],
                'runner_name': row['runner_name'],
                'gender': row['gender'],
                'age': row['age'],
                'country': row['country'],
                'finish_time': row['finish_time'],
                'finish_time_sec': row['finish_time_sec'],
                'segment': seg_name,
                'segment_distance_km': seg_dist,
                'segment_time_sec': int(segment_time),
                'pace_per_km_sec': pace_per_km,
                'segment_type': segment_type,
            })

    result_df = pd.DataFrame(rows)
    return result_df


def main():
    winners_df = generate_winners_dataset()
    winners_df.to_csv(os.path.join(RAW_DIR, 'winners_data.csv'), index=False)

    metadata_df = generate_race_metadata()
    metadata_df.to_csv(os.path.join(RAW_DIR, 'race_metadata.csv'), index=False)

    results_df = generate_marathon_results()
    results_df = add_winner_flags(results_df)
    results_df.to_csv(os.path.join(RAW_DIR, 'marathon_results.csv'), index=False)

    brazilian_df = generate_brazilian_runners(results_df)
    brazilian_df.to_csv(os.path.join(PROCESSED_DIR, 'brazilian_runners_analysis.csv'), index=False)

    pace_splits_df = generate_pace_splits_analysis(results_df)
    pace_splits_df.to_csv(os.path.join(PROCESSED_DIR, 'pace_splits_analysis.csv'), index=False)

    combined_df = results_df.merge(
        metadata_df,
        on=['marathon', 'year'],
        how='left'
    )
    combined_df.to_csv(os.path.join(PROCESSED_DIR, 'combined_marathon_data.csv'), index=False)


if __name__ == '__main__':
    main()
