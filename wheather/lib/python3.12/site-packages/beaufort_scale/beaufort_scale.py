def languages(language):
    dict_languages = {
        'en': {
            "Calm": "Calm",
            "Light_air": "Light air",
            "Light_breeze": "Light breeze",
            "Gentle_breeze": "Gentle breeze",
            "Moderate_breeze": "Moderate breeze",
            "Fresh_breeze": "Fresh breeze",
            "Strong_breeze": "Strong breeze",
            "High_wind": "High wind",
            "Fresh_Gale": "Fresh Gale",
            "Strong_Gale": "Strong Gale",
            "Storm": "Storm",
            "Violent_storm": "Violent storm",
            "Hurricane_force": "Hurricane force"
        },
        'pt-br': {
            "Calm": "Calmo",
            "Light_air": "Aragem",
            "Light_breeze": "Brisa leve",
            "Gentle_breeze": "Brisa fraca",
            "Moderate_breeze": "Brisa moderada",
            "Fresh_breeze": "Brisa forte",
            "Strong_breeze": "Vento fresco",
            "High_wind": "Vento forte",
            "Fresh_Gale": "Ventania",
            "Strong_Gale": "Ventania forte",
            "Storm": "Tempestade",
            "Violent_storm": "Tempestade violenta",
            "Hurricane_force": "Furacão"
        }
    }
    list_languages = list(dict_languages.keys())
    return dict_languages.get(language), list_languages


def loading_language(language):
    #! Clear string
    language = (language.lower()).replace(' ', '')

    #! Test language
    dict_language, list_languages = languages(language)

    if dict_language is not None:
        data = dict_language
    else:
        print(
            f'ERROR: there is no {language} in the translation, select one from the list {list_languages}.'
        )
    return data


def beaufort_scale_ms(value, language='en'):
    #! Loading translate dict
    translate_dict = loading_language(language)

    #! Beaufort scale in m/s
    if(value < 0.5):
        return translate_dict['Calm']
    elif((value >= 0.5) and (value < 1.5)):
        return translate_dict['Light_air']
    elif((value >= 1.5) and (value < 3.3)):
        return translate_dict['Light_breeze']
    elif((value >= 3.3) and (value < 5.5)):
        return translate_dict['Gentle_breeze']
    elif((value >= 5.5) and (value < 7.9)):
        return translate_dict['Moderate_breeze']
    elif((value >= 7.9) and (value < 10.7)):
        return translate_dict['Fresh_breeze']
    elif((value >= 10.7) and (value < 13.8)):
        return translate_dict['Strong_breeze']
    elif((value >= 13.8) and (value < 17.1)):
        return translate_dict['High_wind']
    elif((value >= 17.1) and (value < 20.7)):
        return translate_dict['Fresh_Gale']
    elif((value >= 20.7) and (value < 24.4)):
        return translate_dict['Strong_Gale']
    elif((value >= 24.4) and (value < 28.4)):
        return translate_dict['Storm']
    elif((value >= 28.4) and (value < 32.6)):
        return translate_dict['Violent_storm']
    elif(value >= 32.6):
        return translate_dict['Hurricane_force']


def beaufort_scale_kmh(value, language='en'):
    value_converted = (value/3.6)
    str_value = beaufort_scale_ms(value_converted, language=language)
    return str_value
