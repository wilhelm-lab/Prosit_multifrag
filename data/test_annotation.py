from oktoberfest import preprocessing as pp
import pandas as pd
import os
from pathlib import Path
from spectrum_fundamentals.annotation.annotation import annotate_spectra
from spectrum_fundamentals.fragments import retrieve_ion_types
import oktoberfest as ok
from oktoberfest.data.spectra import Spectra
import numpy as np
from mass_scale import theoretical_ions, my_annotation_function
import yaml
from glob import glob
import re
therm = Path("/cmnfs/data/cluster/software/ThermoRawFileParser/1.4.3/ThermoRawFileParser.exe")
pd.options.mode.chained_assignment = None

{"n[43]": 1, "[160]": 4, "[147]": 35, "[157]": 7, "[129]": 7, "[115]": 7}

with open("yaml/annotate.yaml") as f:
    config = yaml.safe_load(f)
fragmentation_method = config['frag_method']

all_ion_counts = {}
theor_dict = theoretical_ions(
    config['ion_types'], 
    config['max_peptide_length'], 
    config['max_product_charge']
)

        

base_directory1 = "/cmnfs/data/proteomics/shabaz_exotic"
enzymes = [
    m for m in os.listdir(os.path.join(base_directory1, "NAS", fragmentation_method)) 
    if len(m.split('_'))==1
]

all_psms = pd.DataFrame({})
for enzyme in enzymes:
    print((11+len(enzyme))*"#")
    print("# Enzyme %s #"%enzyme)
    print((11+len(enzyme))*"#")

    base_directory = os.path.join(base_directory1, "NAS", fragmentation_method, enzyme)

    ###################
    # convert to mzML #
    ###################

    raw_path = os.path.join(base_directory, 'raw_files')
    mzml_path = os.path.join(base_directory, 'mzml_files')

    list_of_raws = pp.list_spectra(raw_path, input_format='raw')

    # Create the directory if it doesn't exist
    if not os.path.exists(mzml_path):
        os.makedirs(mzml_path)

    # Initialize an empty list to store the paths of the converted mzML files
    conversion_results = []

    # Loop through the list of spectra files and convert each one
    for file in list_of_raws:
        # Generate corresponding mzML file name for each raw file
        base_filename = os.path.splitext(os.path.basename(file))[0]  # Remove file extension
        output_mzml_file = os.path.join(mzml_path, f"{base_filename}.mzML")
        
        # Call the convert_spectra function with the current file and the output file path
        pp.convert_raw_to_mzml(file, output_file=output_mzml_file, thermo_exe=therm)
        
        # Append the path of the converted mzML file to the conversion_results list
        conversion_results.append(output_mzml_file)
        #break

    ################
    # read spectra #
    ################
    mzml_list = pp.list_spectra(mzml_path, input_format='mzml')

    spectra_list = []
    for spectra_file in mzml_list:
        spectra = pp.load_spectra(filenames=spectra_file, parser="pyteomics")
        spectra_list.append(spectra)
        #break

    ###########################
    # read PSM search results #
    ###########################
    merged_search_path = os.path.join(base_directory1, "processed", "merged_search", fragmentation_method)
    if not os.path.exists(os.path.join(merged_search_path, "search")):
        os.makedirs(os.path.join(merged_search_path, "search"))
    
    # Now that the path is created, save your theoretical dictionary
    theor_dict.to_csv(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{fragmentation_method}/all_search_ions.csv", index=True)
    
    # Parse the search results
    #input_path = glob(os.path.join(base_directory, "pepxml", '*'+config['frags_searched'])+'*', )
    input_path = glob(os.path.join(base_directory1, "msfragger_runs/0_pepxml", fragmentation_method, enzyme, "psm.tsv"))
    assert len(input_path) == 1
    #peptides = pp.convert_search(
    #    input_path=input_path[0],
    #    output_file=os.path.join(merged_search_path, "search", f'search.{enzyme}'),
    #    search_engine='MSFragger',
    #)
    peptides = pd.read_csv(input_path[0], sep='\t')
    boolean = peptides['Modified Peptide'].isna()
    peptides['Modified Peptide'][boolean] = pd.Series(peptides['Peptide'][boolean], index=np.where(boolean==True)[0])
    peptides['Modified Peptide'] = peptides['Modified Peptide'].map(lambda x: re.sub('n\[43]','[UNIMOD:1]',x))
    peptides['Modified Peptide'] = peptides['Modified Peptide'].map(lambda x: re.sub('\[160]','[UNIMOD:4]',x))
    peptides['Modified Peptide'] = peptides['Modified Peptide'].map(lambda x: re.sub('\[147]','[UNIMOD:35]',x))
    if config['fixed_cysteine_cam']:
        peptides['Modified Peptide'] = peptides['Modified Peptide'].map(lambda x: re.sub('C','C[UNIMOD:4]',x))
    peptides['RAW_FILE'] = peptides['Spectrum'].map(lambda x: "".join(x.split('.')[:-3]))
    peptides['SCAN_NUMBER'] = peptides['Spectrum'].map(lambda x: int(x.split('.')[-3]))
    peptides['REVERSE'] = pd.Series(len(peptides)*[False], index=peptides.index)
    peptides = peptides.rename(columns={
        "Peptide": "SEQUENCE",
        "Modified Peptide": "MODIFIED_SEQUENCE",
        'Charge': "PRECURSOR_CHARGE",
        'Observed Mass': 'MASS',
        'Hyperscore': 'SCORE',
        'Peptide Length': "PEPTIDE_LENGTH",
        'Mapped Proteins': "PROTEINS",
    })
    drops = [
        'Spectrum', 'Spectrum File', 'Extended Peptide', 'Prev AA', 'Next AA',
        'Retention', 'Calibrated Observed Mass', 'Observed M/Z', 'Calibrated Observed M/Z', 'Calculated Peptide Mass',
        'Calculated M/Z', 'Delta Mass', 'Expectation',
        'Nextscore', 'Number of Enzymatic Termini', 'Number of Missed Cleavages',
        'Protein Start', 'Protein End', 'Intensity', 'Assigned Modifications',
        'Observed Modifications', 'Purity', 'Is Unique', 'Protein',
        'Protein ID', 'Entry Name', 'Gene', 'Protein Description', 'Mapped Genes',
    ]
    if 'PeptideProphet Probability' in peptides.keys():
        drops.append('PeptideProphet Probability')
    if 'Probability' in peptides.keys():
        drops.append('Probability')
    if 'SpectralSim' in peptides.keys():
        drops.append('SpectralSim')
    if 'RTScore' in peptides.keys():
        drops.append('RTScore')
    peptides = peptides.drop(drops, axis=1)

    #############################
    # filter peptides for model #
    #############################
    #filtered_peptides = ok.pp.filter_peptides_for_model(peptides=peptide_df, model='prosit')

    ##########################
    # Merge spectra and PSMs #
    ##########################

    # Ion dictionary
    #ion_types = retrieve_ion_types(fragmentation_method)
    #var_df = Spectra._gen_vars_df(ion_types)
    #var_df.to_csv(os.path.join(merged_search_path, "spectrum_dictionary.csv"))
    psm_list = []
    for spectra in spectra_list:
        psms = pp.merge_spectra_and_peptides(spectra=spectra, search=peptides)
        assert len(psms) > 0, "Merging of search results and spectra didn't work"
        psms['NUM_PEAKS'] = pd.Series(np.vectorize(len)(psms['MZ']), index=psms.index)
        psms['FRAGMENTATION'] = pd.Series(len(psms)*[fragmentation_method], index=psms.index)
        psm_list.append(psms)
    
    ####################
    # Annotate spectra #
    ####################
    for i, psms in enumerate(psm_list):
        print(f"\rAnnotation progress: {i}/{len(psm_list)}", end="")
        #library = pp.annotate_spectral_library(
        #	psms=psms,
        #    fragmentation_method='ECD',
        #	mass_tol=20, 
        #    unit_mass_tol='ppm'
        #)
        #library_list.append(library)
        
        #df_annotated_spectra = annotate_spectra(
        #    un_annot_spectra=psms,
        #    fragmentation_method=fragmentation_method,
        #    mass_tolerance=20,
        #    unit_mass_tolerance='ppm',
        #    annotate_neutral_loss=False,
        #) # ['INTENSITIES', 'MZ', 'CALCULATED_MASS', 'removed_peaks', 'ANNOTATED_NL_COUNT', 'EXPECTED_NL_COUNT']
        #psms['MZ'] = pd.Series([m.astype(np.float32) for m in df_annotated_spectra['MZ']])
        #psms['INTENSITIES'] = pd.Series([m.astype(np.float32) for m in df_annotated_spectra['INTENSITIES']])
        #psms['NUM_ANN_PEAKS'] = pd.Series(np.vectorize(lambda m: sum(m>0))(df_annotated_spectra['INTENSITIES']))
        #library_list.append(df_annotated_spectra)
        

        out_dict = my_annotation_function(psms, theor_dict, threshold_ppm=20, p_window=1.2)
        df = out_dict['dataframe']
        stats = out_dict['statistics']

        # Add annotation data to dataframe
        psms['enzyme'] = pd.Series(psms.shape[0]*[f'{enzyme}'], index=psms.index)
        psms['matched_inds'] = pd.Series([m.astype(np.int32) for m in df['matched_inds']], index=psms.index)
        psms['matched_ions'] = pd.Series(df['matched_ions'])
        psms['matched_ppm'] = pd.Series([m.astype(np.float32) for m in df['matched_ppm']], index=psms.index)
        psms['num_ann_peaks'] = pd.Series(np.vectorize(lambda m: len(m))(df['matched_inds']), index=psms.index)

        # Cast vectors into float32
        psms['MZ'] = psms['MZ'].map(lambda x: x.astype(np.float32))
        psms['INTENSITIES'] = psms['INTENSITIES'].map(lambda x: x.astype(np.float32))

        for ion, count in stats.items():
            if ion not in all_ion_counts:
                all_ion_counts[ion] = 0
            all_ion_counts[ion] += count
    print()
    all_psms = pd.concat([all_psms, pd.concat(psm_list)])

#######################################
# Write merged search results to file #
#######################################

all_psms.index = pd.Index(np.arange(all_psms.shape[0]))

output_path = os.path.join(base_directory1, "processed", "merged_search", fragmentation_method)
all_psms.to_parquet(os.path.join(output_path, f"all_psms.parquet"))

with open(os.path.join(output_path, "ion_counts.tab"), "w") as f:
    all_lines = ["%s\t%d"%(ion_name, count) for ion_name, count in dict(sorted(all_ion_counts.items())).items()]
    f.write("\n".join(all_lines))

