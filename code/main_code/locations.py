import os, sys



def build_location_dict():

    # Enter Dates for save and loading results
    save_date_string = ''
    load_date_string = ''
    file_path = os.path.abspath(__file__)
    main_dir = os.path.dirname(file_path)
    project_dir = '/'.join(main_dir.split('/')[:-2])
    current_save_dir = save_date_string + '_results'

    # Enter saved CSV data name. Do not include path, just filenames.
    ##################################################
    data_csv_name = '' 
    ##################################################
    
    location_dict = {}
    location_dict['load_date'] = load_date_string
    location_dict['save_date'] = save_date_string
    location_dict['project_dir'] = project_dir

    # code dirs
    location_dict['code_dir'] = os.path.join(project_dir, 'code')
    location_dict['main_code'] = os.path.join(location_dict['code_dir'], 'main_code')
    location_dict['jupyter_code'] = os.path.join(location_dict['code_dir'], 'jupyter_code')

    # Data locations and Result directories
    location_dict['result_dir'] = os.path.join(project_dir, 'results')
    location_dict['data_dir'] = os.path.join(location_dict['result_dir'], 'raw_data')
    location_dict['data_csv_loc'] = os.path.join(location_dict['data_dir'], data_csv_name)
    location_dict['save_dir'] = os.path.join(location_dict['result_dir'], current_save_dir)
    location_dict['confusion_dir'] = os.path.join(location_dict['save_dir'], 'confusion_dir')
    location_dict['ROC_dir'] = os.path.join(location_dict['save_dir'], 'ROC_dir')
    location_dict['importance_dir'] = os.path.join(location_dict['save_dir'], 'importance_dir')
    location_dict['excel_results'] = os.path.join(location_dict['save_dir'], 'excel_results')
    location_dict['heatmaps'] = os.path.join(location_dict['save_dir'], 'heatmaps')

    return location_dict




def get_locations(name):
    location_dict = build_location_dict()
    if name in location_dict:
        return location_dict[name]
    else:
        return None