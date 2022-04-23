'''
Contains a routine to find regions of enhanced DFR based on matched DFR_Ku-Ka using
a peak prominence method.

Copyright Joe Finlon, Univ. of Washington, 2022.
'''

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from skimage.measure import label

def find_regions(matched_object, dfr, method='prominances', min_dfr=None, min_prom=2., rel_height=0.4):
    '''
    Inputs:
        matched_object: Dictionary created from matcher routine
        dfr: Masked array of DFR values computed from matched_object
        method: Method for determining enhanced DFR regions/periods ('prominances')
        min_dfr: Minimum DFR to consider for ID scheme (not used for 'prominances' method)
        min_prom: Minimum prominance needed to consider DFR peaks (float)
        rel_height: Relative height at which the peak width is measured as a percentage of its prominence (float between 0 and 1)
    '''
    regions_object = {}
    
    peaks = np.array([], dtype='int'); prominences = np.array([]); width_heights = np.array([])
    durations_p3 = np.array([]); durations_er2 = np.array([])
    peak_starts_p3 = np.array([], dtype='datetime64[ns]'); peak_ends_p3 = np.array([], dtype='datetime64[ns]')
    peak_starts_er2 = np.array([], dtype='datetime64[ns]'); peak_ends_er2 = np.array([], dtype='datetime64[ns]')
    peak_count = 0
    
    labels = label(~dfr.mask) # find contiguious regions/periods where valid (not masked) DFR values exist (peak ID is more robust this way)
    for labelnum in range(1, len(np.unique(labels))+1):
        peaks_temp, _ = find_peaks(dfr[labels==labelnum])
        if len(peaks_temp)>0:
            prominences_temp = peak_prominences(dfr[labels==labelnum], peaks_temp, wlen=None); prominences_temp = prominences_temp[0]
            peaks_temp = peaks_temp[prominences_temp>=min_prom]; prominences_temp = prominences_temp[prominences_temp>=min_prom] # trim peaks and prominences
            widths_temp = peak_widths(dfr[labels==labelnum], peaks_temp, rel_height=rel_height)
            for peaknum in range(len(widths_temp[0])): # loop through each peak to get peak width start/end periods
                peak_count += 1
                width_heights = np.append(width_heights, widths_temp[1][peaknum])
                peak_start_er2 = matched_object['matched']['time_rad']['data'][int(np.where(labels==labelnum)[0][0]+np.floor(widths_temp[2][peaknum]))]
                peak_end_er2 = matched_object['matched']['time_rad']['data'][int(np.where(labels==labelnum)[0][0]+np.ceil(widths_temp[3][peaknum]))]
                peak_start_p3 = matched_object['matched']['time_p3']['data'][int(np.where(labels==labelnum)[0][0]+np.floor(widths_temp[2][peaknum]))]
                peak_end_p3 = matched_object['matched']['time_p3']['data'][int(np.where(labels==labelnum)[0][0]+np.ceil(widths_temp[3][peaknum]))]
                if peak_end_er2<peak_start_er2: # fixes rare instance where peak end needs to be shortened (no matched data after this time)
                    peak_end_er2 = matched_object['matched']['time_rad']['data'][int(np.where(labels==labelnum)[0][0]+np.floor(widths_temp[3][peaknum]))]
                    peak_end_p3 = matched_object['matched']['time_p3']['data'][int(np.where(labels==labelnum)[0][0]+np.floor(widths_temp[3][peaknum]))]
                durations_p3 = np.append(durations_p3, (peak_end_p3-peak_start_p3)/np.timedelta64(1,'s'))
                durations_er2 = np.append(durations_er2, (peak_end_er2-peak_start_er2)/np.timedelta64(1,'s'))
                print('    Peak #{} from {} - {} ({} sec)'.format(peak_count, peak_start_p3, peak_end_p3, durations_p3[-1]))
                peak_starts_p3 = np.append(peak_starts_p3, peak_start_p3); peak_ends_p3 = np.append(peak_ends_p3, peak_end_p3)
                peak_starts_er2 = np.append(peak_starts_er2, peak_start_er2); peak_ends_er2 = np.append(peak_ends_er2, peak_end_er2)

            peaks = np.append(peaks, np.where(labels==labelnum)[0][0]+peaks_temp)
            prominences = np.append(prominences, prominences_temp)
            
    # Construct the object
    regions_object['peak_start_p3'] = peak_starts_p3; regions_object['peak_end_p3'] = peak_ends_p3
    regions_object['peak_start_er2'] = peak_starts_er2; regions_object['peak_end_er2'] = peak_ends_er2
    regions_object['width_height'] = width_heights # height of the contour lines at which the widths where evaluated
    regions_object['peak_index'] = peaks; regions_object['peak_value'] = dfr[peaks]; regions_object['peak_prominence'] = prominences
    regions_object['duration_p3'] = durations_p3; regions_object['duration_er2'] = durations_er2
    regions_object['stats'] = {}
    regions_object['stats']['num_regions'] = peak_count
    regions_object['stats']['mean_duration_p3'] = np.sum(durations_p3) / peak_count
    regions_object['stats']['mean_duration_er2'] = np.sum(durations_er2) / peak_count
    
    return regions_object