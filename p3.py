'''
Contains routines to read and manipulate particle size distribution data from the
2D-S and HVPS optical array probes during the IMPACTS experiment. Creation of 1 Hz
PSD data requires use of UIOOPS package (doi: 10.5281/zenodo.3976291).
Some of the arguments in the routines require radar data matched to the P-3
location to determine the optional degree of riming following Leinonen & Szyrmer (2015).

Copyright Joe Finlon, Univ. of Washington, 2022.
'''

import xarray as xr
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.warnings.filterwarnings('ignore', message='Mean of empty slice')
np.seterr(invalid='ignore')
from datetime import datetime, timedelta
from scipy.optimize import least_squares
try: # try importing the pytmatrix package
    from forward import *
except ImportError:
    print(
        'WARNING: The pytmatrix package cannot be installed for the psdread() function.'
    )

    
def psdread(
        twodsfile, hvpsfile, datestr, size_cutoff=1., minD=0.15, maxD=30.,
        qc=False, deadtime_thresh=0.6, verbose=True,
        start_time=None, end_time=None, tres=5.,
        compute_bulk=False, compute_fits=False, Z_interp=False,
        matchedZ_W=None, matchedZ_Ka=None, matchedZ_Ku=None, matchedZ_X=None):
    '''
    Load the 2DS and HVPS PSDs processed by UIOOPS and create n-second combined PSDs with optional bulk properties.
    Inputs:
        twodsfile: Path to the 2DS data
        hvpsfile: Path to the HVPS data
        datestr: YYYYMMDD [str]
        size_cutoff: Size [mm] for the 2DS-HVPS crossover
        minD: Minimum size [mm] to consider in the combined PSD
        maxD: Maximum size [mm] to consider in the combined PSD
        qc: Boolean to optionally ignore 1-Hz data from averaging when probe dead time > deadtime_thresh
        deadtime_thresh: Deadtime hreshold [0–1] to ignore 1-Hz data when qc is True
        verbose: Boolean to optionally print all data-related warnings (e.g., high probe dead time)
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
        compute_bulk: Boolean to optionally compute bulk statistics such as N, IWC, Dmm, rho_e
        compute_fits: Boolean to optionally compute gamma fit parameters N0, mu, lambda
        Z_interp: Boolean to optionally simulate Z for additional degrees of riming from Leinonen & Szyrmer (2015; LS15)
        matchedZ_Ka, ...: None (skips minimization) or masked array of matched Z values to perform LS15 m-D minimization
    '''
    p3psd = {}

    if (twodsfile is None) and (hvpsfile is not None):
        print('Only using the HVPS data for {}'.format(datestr))
        size_cutoff = 0.4 # start HVPS PSD at 0.4 mm
    elif (hvpsfile is None) and (twodsfile is not None):
        print('Only using the 2DS data for {}'.format(datestr))
        size_cutoff = 3.2 # end 2DS PSD at 3.2 mm
    elif (twodsfile is None) and (hvpsfile is None):
        print('No input files given...exiting')
        exit()

    # 2DS information
    if twodsfile is not None:
        ds1 = xr.open_dataset(twodsfile)
        time_raw = ds1['time'].values # HHMMSS from flight start date or numpy.datetime64
        if np.issubdtype(time_raw.dtype, np.datetime64): # numpy.datetime64
            time = np.array(time_raw, dtype='datetime64[s]')
        else: # native HHMMSS format (from UIOOPS SD file)
            time_dt = [
                datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:]))
                + timedelta(
                    hours=int(str(int(time_raw[i])).zfill(6)[0:2]),
                    minutes=int(str(int(time_raw[i])).zfill(6)[2:4]),
                    seconds=int(str(int(time_raw[i])).zfill(6)[4:]))
                for i in range(len(time_hhmmss))
            ]
            time_str = [
                datetime.strftime(time_dt[i], '%Y-%m-%dT%H:%M:%S')
                for i in range(len(time_dt))
            ]
            time = np.array(time_str, dtype='datetime64[s]')
        bin_min_2ds = ds1['bin_min'].values # mm
        bin_max_2ds = ds1['bin_max'].values
        bin_inds = np.where((bin_min_2ds>=minD) & (bin_max_2ds<=size_cutoff))[0] # find bins within user-specified range
        bin_min_2ds = bin_min_2ds[bin_inds]; bin_max_2ds = bin_max_2ds[bin_inds]
        bin_width_2ds = ds1['bin_dD'].values[bin_inds] / 10. # cm
        bin_mid_2ds = bin_min_2ds + (bin_width_2ds * 10.) / 2.
        count_2ds = ds1['count'].values[:, bin_inds]
        sv_2ds = ds1['sample_vol'].values[:, bin_inds] # cm^3
        count_hab_2ds = ds1['habitsd'].values[:, bin_inds, :] * np.tile(np.reshape(sv_2ds, (sv_2ds.shape[0], sv_2ds.shape[1], 1)), (1, 1, 10)) * np.tile(
            np.reshape(bin_width_2ds, (1, len(bin_width_2ds), 1)), (sv_2ds.shape[0], 1, 10))
        ar_2ds = ds1['mean_area_ratio'].values[:, bin_inds] # mean area ratio (circular fit) per bin
        asr_2ds = ds1['mean_aspect_ratio_ellipse'].values[:, bin_inds] # mean aspect ratio (elliptical fit) per bin
        activetime_2ds = ds1['sum_IntArr'].values # s

        if hvpsfile is None:
            count = count_2ds; count_hab = count_hab_2ds; sv = sv_2ds; ar = ar_2ds; asr = asr_2ds; activetime_hvps = np.ones(count.shape[0])
            bin_min = bin_min_2ds; bin_mid = bin_mid_2ds; bin_max = bin_max_2ds; bin_width = bin_width_2ds

    # HVPS information
    if hvpsfile is not None:
        ds2 = xr.open_dataset(hvpsfile)
        bin_min_hvps = ds2['bin_min'].values # mm
        bin_max_hvps = ds2['bin_max'].values
        bin_inds = np.where((bin_min_hvps>=size_cutoff) & (bin_max_hvps<=maxD))[0] # find bins within user-specified range
        bin_min_hvps = bin_min_hvps[bin_inds]; bin_max_hvps = bin_max_hvps[bin_inds]
        bin_width_hvps = ds2['bin_dD'].values[bin_inds] / 10. # cm
        if size_cutoff==2.:
            bin_min_hvps = np.insert(bin_min_hvps, 0, 2.); bin_max_hvps = np.insert(bin_max_hvps, 0, 2.2); bin_width_hvps = np.insert(bin_width_hvps, 0, 0.02)
            bin_inds = np.insert(bin_inds, 0, bin_inds[0]-1)
        bin_mid_hvps = bin_min_hvps + (bin_width_hvps * 10.) / 2.
        count_hvps = ds2['count'].values[:, bin_inds]
        sv_hvps = ds2['sample_vol'].values[:, bin_inds] # cm^3
        count_hab_hvps = (ds2['habitsd'].values[:, bin_inds, :]) * np.tile(np.reshape(sv_hvps, (sv_hvps.shape[0], sv_hvps.shape[1], 1)), (1, 1, 10)) * np.tile(
            np.reshape(bin_width_hvps, (1, len(bin_width_hvps), 1)), (sv_hvps.shape[0], 1, 10))
        ar_hvps = ds2['mean_area_ratio'].values[:, bin_inds] # mean area ratio (circular fit) per bin
        asr_hvps = ds2['mean_aspect_ratio_ellipse'].values[:, bin_inds] # mean aspect ratio (elliptical fit) per bin
        activetime_hvps = ds2['sum_IntArr'].values # s
        if size_cutoff==2.: # normalize counts in first bin (1.8-2.2 mm, now only for 2-2.2 mm)
            count_hvps[:, 0] = count_hvps[:, 0] / 2.
            count_hab_hvps[:, 0, :] = count_hab_hvps[:, 0, :] / 2.

        if twodsfile is None:
            time_hhmmss = ds2['time'].values # HHMMSS from flight start date
            time_dt = [datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:])) + timedelta(
                hours=int(str(int(time_hhmmss[i])).zfill(6)[0:2]), minutes=int(str(int(time_hhmmss[i])).zfill(6)[2:4]),
                seconds=int(str(int(time_hhmmss[i])).zfill(6)[4:])) for i in range(len(time_hhmmss))]
            time_str = [datetime.strftime(time_dt[i], '%Y-%m-%dT%H:%M:%S') for i in range(len(time_dt))]
            time = np.array(time_str, dtype='datetime64[s]')
            count = count_hvps; count_hab = count_hab_hvps; sv = sv_hvps; ar = ar_hvps; asr = asr_hvps; activetime_2ds = np.ones(count.shape[0])
            bin_min = bin_min_hvps; bin_mid = bin_mid_hvps; bin_max = bin_max_hvps; bin_width = bin_width_hvps

    # Combine the datasets
    if (twodsfile is not None) and (hvpsfile is not None):
        count = np.concatenate((count_2ds, count_hvps), axis=1)
        count_hab = np.concatenate((count_hab_2ds, count_hab_hvps), axis=1)
        sv = np.concatenate((sv_2ds, sv_hvps), axis=1)
        ar = np.concatenate((ar_2ds, ar_hvps), axis=1)
        asr = np.concatenate((asr_2ds, asr_hvps), axis=1)
        bin_min = np.concatenate((bin_min_2ds, bin_min_hvps))
        bin_mid = np.concatenate((bin_mid_2ds, bin_mid_hvps))
        bin_max = np.concatenate((bin_max_2ds, bin_max_hvps))
        bin_width = np.concatenate((bin_width_2ds, bin_width_hvps))

    # Average the data
    if start_time is None:
        start_dt64 = time[0]
    else:
        start_dt64 = np.datetime64(start_time)
    if end_time is None:
        end_dt64 = time[-1] if int(tres)>1 else time[-1]+np.timedelta64(1, 's')
    else:
        end_dt64 = np.datetime64(end_time) if int(tres)>1 else np.datetime64(end_time)+np.timedelta64(1, 's')
    dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

    # Allocate arrays
    count_aver = np.zeros((int(dur/tres), len(bin_mid)))
    count_hab_aver = np.zeros((int(dur/tres), len(bin_mid), 8))
    sv_aver = np.zeros((int(dur/tres), len(bin_mid)))
    at_2ds_aver = np.ma.array(np.ones(int(dur/tres)), mask=False)
    at_hvps_aver = np.ma.array(np.ones(int(dur/tres)), mask=False)
    ND = np.zeros((int(dur/tres), len(bin_mid)))
    ar_aver = np.zeros((int(dur/tres), len(bin_mid)))
    asr_aver = np.zeros((int(dur/tres), len(bin_mid)))

    time_subset = start_dt64 # allocate time array of N-sec interval obs
    curr_time = start_dt64
    i = 0

    while curr_time+np.timedelta64(int(tres),'s')<=end_dt64:
        if curr_time>start_dt64:
            time_subset = np.append(time_subset, curr_time)
        time_inds = np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]
        if qc is True:
            activetime_thresh = 1. - deadtime_thresh
            time_inds = time_inds[(activetime_2ds[time_inds]>=activetime_thresh) & (activetime_hvps[time_inds]>=activetime_thresh)]
        if len(time_inds)>0:
            count_aver[i, :] = np.nansum(count[time_inds, :], axis=0)
            count_hab_aver[i, :, 0] = np.nansum(count_hab[time_inds, :, 3], axis=0) # tiny
            count_hab_aver[i, :, 1] = np.nansum(count_hab[time_inds, :, 0], axis=0) # spherical
            count_hab_aver[i, :, 2] = np.nansum(count_hab[time_inds, :, 1:3], axis=(0, 2)) # oriented + linear
            count_hab_aver[i, :, 3] = np.nansum(count_hab[time_inds, :, 4], axis=0) # hexagonal
            count_hab_aver[i, :, 4] = np.nansum(count_hab[time_inds, :, 5], axis=0) # irregular
            count_hab_aver[i, :, 5] = np.nansum(count_hab[time_inds, :, 6], axis=0) # graupel
            count_hab_aver[i, :, 6] = np.nansum(count_hab[time_inds, :, 7], axis=0) # dendrite
            count_hab_aver[i, :, 7] = np.nansum(count_hab[time_inds, :, 8], axis=0) # aggregate
            ar_aver[i, :] = np.nanmean(ar[time_inds, :], axis=0) # binned mean of area ratio
            asr_aver[i, :] = np.nanmean(asr[time_inds, :], axis=0) # binned mean of aspect ratio
            sv_aver[i, :] = np.nansum(sv[time_inds, :], axis=0)
            at_2ds_aver[i] = np.nansum(activetime_2ds[time_inds]) / len(time_inds)
            at_hvps_aver[i] = np.nansum(activetime_hvps[time_inds]) / len(time_inds)
            ND[i, :] = np.nanmean(count[time_inds, :]/sv[time_inds, :], axis=0) / bin_width # take N(D) for each sec, then average [cm**-4]
        else: # Mask data for current period if dead (active) time from either probe > 0.8*tres (< 0.2*tres) for all 1-Hz times
            if verbose is True:
                print('All 1-Hz data for the {}-s period beginning {} has high dead time. Masking data.'.format(str(tres), np.datetime_as_string(curr_time)))
            at_2ds_aver[i] = np.nansum(activetime_2ds[np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]]) / tres; at_2ds_aver.mask[i] = True
            at_hvps_aver[i] = np.nansum(activetime_hvps[np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]]) / tres; at_hvps_aver.mask[i] = True
            count_aver[i, :] = np.nan; count_hab_aver[i, :] = np.nan; sv_aver[i, :] = np.nan; ND[i, :] = np.nan; asr_aver[i, :] = np.nan
        i += 1
        curr_time += np.timedelta64(int(tres), 's')

    #ND = np.ma.masked_invalid(count_aver / sv_aver / np.tile(bin_width[np.newaxis, :], (int(dur/tres), 1))) # cm^-4

    # Mask arrays
    count_aver = np.ma.masked_where(np.isnan(count_aver), count_aver)
    count_hab_aver = np.ma.masked_where(np.isnan(count_hab_aver), count_hab_aver)
    sv_aver = np.ma.masked_where(np.isnan(sv_aver), sv_aver)
    ar_aver = np.ma.masked_invalid(ar_aver)
    asr_aver = np.ma.masked_invalid(asr_aver)
    ND[~np.isfinite(ND)] = 0.; ND = np.ma.masked_where(ND==0., ND)

    # Create dictionary
    p3psd['time'] = time_subset
    p3psd['count'] = count_aver
    p3psd['count_habit'] = count_hab_aver
    p3psd['sv'] = sv_aver
    p3psd['area_ratio'] = ar_aver
    p3psd['aspect_ratio'] = asr_aver
    p3psd['ND'] = ND
    p3psd['bin_min'] = bin_min
    p3psd['bin_mid'] = bin_mid
    p3psd['bin_max'] = bin_max
    p3psd['bin_width'] = bin_width
    p3psd['active_time_2ds'] = at_2ds_aver
    p3psd['active_time_hvps'] = at_hvps_aver

    if compute_bulk is True:
        # Compute Z for various degrees of riming and radar wavelengths
        # Based on work from Leionen and Szyrmer 2015 (LS15)
        # (https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2015EA000102)
        # Follows https://github.com/dopplerchase/Leinonen_Python_Forward_Model
        # and uses forward.py and ess238-sup-0002-supinfo.tex in repo
        Z = forward_Z() #initialize class
        # get the PSD in the format to use in the routine (mks units)
        Z.set_PSD(PSD=ND*10.**8, D=bin_mid/1000., dD=bin_width/100., Z_interp=Z_interp)
        Z.load_split_L15() # Load the leinonen output
        Z.fit_sigmas(Z_interp) # Fit the backscatter cross-sections
        Z.fit_rimefrac(Z_interp) # Fit the riming fractions
        Z.calc_Z() # Calculate Z...outputs are Z.Z_x, Z.Z_ku, Z.Z_ka, Z.Z_w for the four radar wavelengths

        # Compute IWC and Dmm following Brown and Francis (1995), modified for a Dmax definition following Hogan et al.
        [
            N0_bf, N0_hy, mu_bf, mu_hy, lam_bf, lam_hy, iwc_bf, iwc_hy, iwc_hab,
            asr_nw, asr_bf, asr_hy, asr_hab, dmm_bf, dmm_hy, dmm_hab, dm_bf, dm_hy,
            dm_hab, rho_bf, rho_hy, rho_hab, rhoe_bf, rhoe_hy, rhoe_hab] = calc_bulk(
            count_aver, count_hab_aver, sv_aver, asr_aver, bin_mid, bin_width)

        # Add bulk variables to the dictionary
        if Z_interp is True: # consider additional degrees of riming from LS15
            p3psd['riming_mass_array'] = [
                0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 2.]
        else:
            p3psd['riming_mass_array'] = [0., 0.1, 0.2, 0.5, 1., 2.]
        p3psd['a_coeff_array'] = Z.a_coeff
        p3psd['b_coeff_array'] = Z.b_coeff
        p3psd['dbz_W'] = Z.Z_w
        p3psd['dbz_Ka'] = Z.Z_ka
        p3psd['dbz_Ku'] = Z.Z_ku
        p3psd['dbz_X'] = Z.Z_x
        p3psd['N0_bf'] = N0_bf
        p3psd['N0_hy'] = N0_hy
        p3psd['mu_bf'] = mu_bf
        p3psd['mu_hy'] = mu_hy
        p3psd['lambda_bf'] = lam_bf
        p3psd['lambda_hy'] = lam_hy
        p3psd['iwc_bf'] = iwc_bf
        p3psd['iwc_hy'] = iwc_hy
        p3psd['iwc_hab'] = iwc_hab
        p3psd['mean_aspect_ratio'] = asr_nw
        p3psd['mean_aspect_ratio_bf'] = asr_bf
        p3psd['mean_aspect_ratio_hy'] = asr_hy
        p3psd['mean_aspect_ratio_habit'] = asr_hab
        p3psd['dmm_bf'] = dmm_bf
        p3psd['dmm_hy'] = dmm_hy
        p3psd['dmm_hab'] = dmm_hab
        p3psd['dm_bf'] = dm_bf
        p3psd['dm_hy'] = dm_hy
        p3psd['dm_hab'] = dm_hab
        p3psd['eff_density_bf'] = rhoe_bf
        p3psd['eff_density_hy'] = rhoe_hy
        p3psd['eff_density_hab'] = rhoe_hab
        p3psd['density_bf'] = rho_bf
        p3psd['density_hy'] = rho_hy
        p3psd['density_hab'] = rho_hab

        # Optionally constrain the matched Z at Ku- and Ka-band against PSDS to estimate bulk properties
        if (
                matchedZ_W is not None) or (matchedZ_Ka is not None) or (
                matchedZ_Ku is not None) or (matchedZ_X is not None):
            p3psd = calc_riming(
                p3psd, Z, matchedZ_W, matchedZ_Ka, matchedZ_Ku, matchedZ_X,
                compute_fits=compute_fits)

    return p3psd

def calc_bulk(particle_count, habit_count, sample_vol, aspect_ratio, bin_mid, bin_width):
    x0 = [1.e-1, -1., 5.] # initial guess for N0 [cm**-4], mu, lambda [cm**-1]

    # allocate arrays
    N0_bf = np.zeros(particle_count.shape[0])
    N0_hy = np.zeros(particle_count.shape[0])
    mu_bf = np.zeros(particle_count.shape[0])
    mu_hy = np.zeros(particle_count.shape[0])
    lam_bf = np.zeros(particle_count.shape[0])
    lam_hy = np.zeros(particle_count.shape[0])
    iwc_bf = np.zeros(particle_count.shape[0])
    iwc_hy = np.zeros(particle_count.shape[0])
    iwc_hab = np.zeros(particle_count.shape[0])
    asr_nw = np.zeros(particle_count.shape[0])
    asr_bf = np.zeros(particle_count.shape[0])
    asr_hy = np.zeros(particle_count.shape[0])
    asr_hab = np.zeros(particle_count.shape[0])
    dmm_bf = np.zeros(particle_count.shape[0])
    dmm_hy = np.zeros(particle_count.shape[0])
    dmm_hab = np.zeros(particle_count.shape[0])
    dm_bf = np.zeros(particle_count.shape[0])
    dm_hy = np.zeros(particle_count.shape[0])
    dm_hab = np.zeros(particle_count.shape[0])
    rhoe_bf = np.zeros(particle_count.shape[0])
    rhoe_hy = np.zeros(particle_count.shape[0])
    rhoe_hab = np.zeros(particle_count.shape[0])
    rho_bf = np.zeros((particle_count.shape[0], particle_count.shape[1]))
    rho_hy = np.zeros((particle_count.shape[0], particle_count.shape[1]))
    rho_hab = np.zeros((particle_count.shape[0], particle_count.shape[1]))

    # compute particle habit mass outside loop for speed
    a_coeff = np.array([1.96e-3, 1.96e-3, 1.666e-3, 7.39e-3, 1.96e-3, 4.9e-2, 5.16e-4, 1.96e-3])
    a_tile = np.tile(np.reshape(a_coeff, (1, len(a_coeff))), (habit_count.shape[1], 1))
    b_coeff = np.array([1.9, 1.9, 1.91, 2.45, 1.9, 2.8, 1.8, 1.9])
    b_tile = np.tile(np.reshape(b_coeff, (1, len(b_coeff))), (habit_count.shape[1], 1))
    D_tile = np.tile(np.reshape(bin_mid, (len(bin_mid), 1)), (1, habit_count.shape[2]))
    mass_tile = a_tile * (D_tile/10.) ** b_tile

    for time_ind in range(particle_count.shape[0]):
        if particle_count[time_ind, :].count()==particle_count.shape[1]: # time period is not masked...continue on
            Nt = 1000.*np.nansum(particle_count[time_ind, :]/sample_vol[time_ind, :]) # number concentratino [L**-1]

            # spherical volume from Chase et al. (2018) [cm**3 / cm**3]
            vol = (np.pi / 6.) * np.sum(0.6 * ((bin_mid/10.)**3.) * particle_count[time_ind, :] / sample_vol[time_ind, :])

            # number-weighted mean aspect rato
            asr_nw[time_ind] = np.nansum(aspect_ratio[time_ind, :] * particle_count[time_ind, :]) / np.nansum(particle_count[time_ind, :])

            # Brown & Francis products
            mass_particle = (0.00294/1.5) * (bin_mid/10.)**1.9 # particle mass [g]
            mass_bf = mass_particle * particle_count[time_ind, :] # g (binned)
            cumMass_bf = np.nancumsum(mass_bf)
            if cumMass_bf[-1]>0.:
                iwc_bf[time_ind] = 10.**6 * np.nansum(mass_bf / sample_vol[time_ind, :]) # g m^-3
                z_bf = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*particle_count[time_ind, :]/sample_vol[time_ind, :]) # mm^6 m^-3
                sol = least_squares(calc_chisquare, x0, method='lm',ftol=1e-9,xtol=1e-9, max_nfev=int(1e6),\
                                    args=(Nt,iwc_bf[time_ind],z_bf,bin_mid,bin_width,0.00294/1.5,1.9)) # sove the gamma params using least squares minimziation
                N0_bf[time_ind] = sol.x[0]; mu_bf[time_ind] = sol.x[1]; lam_bf[time_ind] = sol.x[2]
                asr_bf[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_bf / sample_vol[time_ind, :]) / np.sum(mass_bf / sample_vol[time_ind, :]) # mass-weighted aspect ratio
                rhoe_bf[time_ind] = (iwc_bf[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
                rho_bf[time_ind, :] = (mass_bf / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
                dm_bf[time_ind] = 10. * np.sum((bin_mid/10.) * mass_bf / sample_vol[time_ind, :]) / np.sum(mass_bf / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]
                if cumMass_bf[0]>=0.5*cumMass_bf[-1]:
                    dmm_bf[time_ind] = bin_mid[0]
                else:
                    dmm_bf[time_ind] = bin_mid[np.where(cumMass_bf>0.5*cumMass_bf[-1])[0][0]-1]

            # Heymsfield (2010) products [https://doi.org/10.1175/2010JAS3507.1]
            #mass_hy = (0.0061*(bin_mid/10.)**2.05) * particle_count[time_ind, :] # g (binned) H04 definition used in GPM NCAR files
            mass_particle = 0.00528 * (bin_mid/10.)**2.1 # particle mass [g]
            mass_hy = mass_particle * particle_count[time_ind, :] # g (binned)
            cumMass_hy = np.nancumsum(mass_hy)
            if cumMass_hy[-1]>0.:
                iwc_hy[time_ind] = 10.**6 * np.nansum(mass_hy / sample_vol[time_ind, :]) # g m^-3
                z_hy = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*particle_count[time_ind, :]/sample_vol[time_ind, :]) # mm^6 m^-3
                sol = least_squares(calc_chisquare, x0, method='lm',ftol=1e-9,xtol=1e-9, max_nfev=int(1e6),\
                                    args=(Nt,iwc_hy[time_ind],z_hy,bin_mid,bin_width,0.00528,2.1)) # sove the gamma params using least squares minimziation
                N0_hy[time_ind] = sol.x[0]; mu_hy[time_ind] = sol.x[1]; lam_hy[time_ind] = sol.x[2]
                asr_hy[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_hy / sample_vol[time_ind, :]) / np.sum(mass_hy / sample_vol[time_ind, :]) # mass-weighted aspect ratio
                rhoe_hy[time_ind] = (iwc_hy[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
                rho_hy[time_ind, :] = (mass_hy / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
                dm_hy[time_ind] = 10. * np.sum((bin_mid/10.) * mass_hy / sample_vol[time_ind, :]) / np.sum(mass_hy / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]
                if cumMass_hy[0]>=0.5*cumMass_hy[-1]:
                    dmm_hy[time_ind] = bin_mid[0]
                else:
                    dmm_hy[time_ind] = bin_mid[np.where(cumMass_hy>0.5*cumMass_hy[-1])[0][0]-1]


            # Habit-specific products
            mass_hab = np.sum(mass_tile * habit_count[time_ind, :, :], axis=1) # g (binned)
            cumMass_hab = np.nancumsum(mass_hab)
            if cumMass_hab[-1]>0.:
                if cumMass_hab[0]>=0.5*cumMass_hab[-1]:
                    dmm_hab[time_ind] = bin_mid[0]
                else:
                    dmm_hab[time_ind] = bin_mid[np.where(cumMass_hab>0.5*cumMass_hab[-1])[0][0]-1]
            iwc_hab[time_ind] = 10.**6 * np.nansum(mass_hab / sample_vol[time_ind, :]) # g m^-3
            asr_hab[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_hab / sample_vol[time_ind, :]) / np.sum(mass_hab / sample_vol[time_ind, :]) # mass-weighted aspect ratio
            rhoe_hab[time_ind] = (iwc_hab[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
            rho_hab[time_ind, :] = (mass_hab / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
            dm_hab[time_ind] = 10. * np.sum((bin_mid/10.) * mass_hab / sample_vol[time_ind, :]) / np.sum(mass_hab / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]

    mu_bf = np.ma.masked_where(N0_bf==0., mu_bf)
    mu_hy = np.ma.masked_where(N0_hy==0., mu_hy)
    lam_bf = np.ma.masked_where(N0_bf==0., lam_bf)
    lam_hy = np.ma.masked_where(N0_hy==0., lam_hy)
    N0_bf = np.ma.masked_where(N0_bf==0., N0_bf)
    N0_hy = np.ma.masked_where(N0_hy==0., N0_hy)
    dmm_bf = np.ma.masked_where(dmm_bf==0., dmm_bf)
    dmm_hy = np.ma.masked_where(dmm_hy==0., dmm_hy)
    dmm_hab = np.ma.masked_where(dmm_hab==0., dmm_hab)
    dm_bf = np.ma.masked_where(dm_bf==0., dm_bf)
    dm_hy = np.ma.masked_where(dm_hy==0., dm_hy)
    dm_hab = np.ma.masked_where(dm_hab==0., dm_hab)
    asr_nw = np.ma.masked_where(np.ma.getmask(dmm_bf), asr_nw)
    asr_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), asr_bf)
    asr_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), asr_hy)
    asr_hab = np.ma.masked_where(np.ma.getmask(asr_hab), iwc_hab)
    rhoe_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), rhoe_bf)
    rhoe_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), rhoe_hy)
    rhoe_hab = np.ma.masked_where(np.ma.getmask(dmm_hab), rhoe_hab)
    iwc_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), iwc_bf)
    iwc_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), iwc_hy)
    iwc_hab = np.ma.masked_where(np.ma.getmask(dmm_hab), iwc_hab)
    rho_bf = np.ma.masked_where(rho_bf==0., rho_bf)
    rho_hy = np.ma.masked_where(rho_hy==0., rho_hy)
    rho_hab = np.ma.masked_where(rho_hab==0., rho_hab)

    return (N0_bf, N0_hy, mu_bf, mu_hy, lam_bf, lam_hy, iwc_bf, iwc_hy, iwc_hab, asr_nw, asr_bf, asr_hy, asr_hab, dmm_bf, dmm_hy, dmm_hab,\
            dm_bf, dm_hy, dm_hab, rho_bf, rho_hy, rho_hab, rhoe_bf, rhoe_hy, rhoe_hab)

def calc_riming(p3psd, Z, matchedZ_W, matchedZ_Ka, matchedZ_Ku, matchedZ_X, compute_fits=False):
    x0 = [1.e-1, -1., 5.] # initial guess for N0 [cm**-4], mu, lambda [cm**-1]

    rmass = np.zeros(len(p3psd['time']))
    rfrac = np.zeros(len(p3psd['time']))
    a_coeff = np.zeros(len(p3psd['time']))
    b_coeff = np.zeros(len(p3psd['time']))
    Nw = np.zeros(len(p3psd['time']))
    N0 = np.zeros(len(p3psd['time']))
    mu = np.zeros(len(p3psd['time']))
    lam = np.zeros(len(p3psd['time']))
    iwc = np.zeros(len(p3psd['time']))
    asr = np.zeros(len(p3psd['time']))
    dm = np.zeros(len(p3psd['time']))
    dmm = np.zeros(len(p3psd['time']))
    rho_eff = np.zeros(len(p3psd['time']))
    dfr_KuKa = np.zeros(len(p3psd['time']))
    error = np.zeros((len(p3psd['time']), len(p3psd['riming_mass_array'])))

    for i in range(len(p3psd['time'])):
        # loop through the different possible riming masses
        for j in range(len(p3psd['riming_mass_array'])):
            if (matchedZ_W is not None) and (np.ma.is_masked(matchedZ_W[i]) is False) and (np.ma.is_masked(p3psd['dbz_W'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_W[i] - p3psd['dbz_W'][i, j])
            if (matchedZ_Ka is not None) and (np.ma.is_masked(matchedZ_Ka[i]) is False) and (np.ma.is_masked(p3psd['dbz_Ka'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_Ka[i] - p3psd['dbz_Ka'][i, j])
            if (matchedZ_Ku is not None) and (np.ma.is_masked(matchedZ_Ku[i]) is False) and (np.ma.is_masked(p3psd['dbz_Ku'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_Ku[i] - p3psd['dbz_Ku'][i, j])
            if (matchedZ_X is not None) and (np.ma.is_masked(matchedZ_X[i]) is False) and (np.ma.is_masked(p3psd['dbz_X'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_X[i] - p3psd['dbz_X'][i, j])

        if np.sum(error[i, :])>0.:
            rmass[i] = p3psd['riming_mass_array'][np.argmin(error[i, :])]
            a_coeff[i] = p3psd['a_coeff_array'][np.argmin(error[i, :])]
            b_coeff[i] = p3psd['b_coeff_array'][np.argmin(error[i, :])]

            if p3psd['count'][i, :].count()==p3psd['count'].shape[1]: # time period is not masked...continue on
                Nt = 1000.*np.nansum(p3psd['count'][i, :]/p3psd['sv'][i, :]) # concentration [L**-1]
                mass_particle = a_coeff[i] * (p3psd['bin_mid']/10.)**b_coeff[i] # particle mass [g]
                mass = mass_particle * p3psd['count'][i, :] # g (binned)
                cumMass = np.nancumsum(mass)
                if cumMass[-1]>0.:
                    # Nw (follows Chase et al. 2021)
                    # [log10(m**-3 mm**-1)]
                    D_melt = ((6. * mass_particle) / (np.pi * 0.997))**(1./3.)
                    Nw[i] = np.log10((1e5) * (4.**4 / 6) * np.nansum(
                        D_melt**3 * p3psd['ND'][i, :] * p3psd['bin_width'])**5 / np.nansum(
                        D_melt**4 * p3psd['ND'][i, :] * p3psd['bin_width'])**4)

                    # IWC
                    iwc[i] = 10.**6 * np.nansum(mass / p3psd['sv'][i, :]) # g m^-3

                    # DFR
                    dfr_KuKa[i] = p3psd[
                        'dbz_Ku'][i, np.argmin(error[i, :])] - p3psd[
                        'dbz_Ka'][i, np.argmin(error[i, :])] # dB

                    # Optionally compute N0, mu, lambda
                    if compute_fits:
                        z = 10.**(p3psd['dbz_X'][i,np.argmin(error[i, :])]/10.) # mm^6 m^-3

                        # solve gamma params using least squares minimziation
                        sol = least_squares(
                            calc_chisquare, x0, method='lm', ftol=1e-9, xtol=1e-9,
                            max_nfev=int(1e6), args=(
                                Nt, iwc[i], z, p3psd['bin_mid'], p3psd['bin_width'],
                                a_coeff[i], b_coeff[i], np.argmin(error[i, :])))
                        N0[i] = sol.x[0]; mu[i] = sol.x[1]; lam[i] = sol.x[2]

                    # Mass-weighted mean aspect ratio
                    asr[i] = np.sum(
                        p3psd['aspect_ratio'][i, :] * mass / p3psd['sv'][i, :]) / np.sum(
                        mass / p3psd['sv'][i, :])

                    # Bulk riming fraction (see Eqn 1 of Morrison and Grabowski
                    # [2010, https://doi.org/10.1175/2010JAS3250.1] for binned version)
                    rfrac[i] = np.sum(
                        np.squeeze(Z.rimefrac[0, :, np.argmin(error[i, :])])
                        * mass / p3psd['sv'][i, :]) / np.nansum(
                        mass / p3psd['sv'][i, :]) # SUM(rimed mass conc)/iwc

                    # Effective density (follows Chase et al. 2018)
                    vol = (np.pi / 6.) * np.sum(
                        0.6 * ((p3psd['bin_mid']/10.)**3.) * p3psd['count'][i, :]
                        / p3psd['sv'][i, :]) # [cm**3 / cm**3]
                    rho_eff[i] = (iwc[i] / 10.**6) / vol # [g cm**-3]

                    # Mass-weighted mean diameter (follows Chase et al. 2020)
                    # M3/M2 if b==2, more generally M(b+1)/Mb
                    dm[i] = 10. * np.sum(
                        (p3psd['bin_mid']/10.) * mass / p3psd['sv'][i, :]) / np.sum(
                        mass / p3psd['sv'][i, :]) # [mm]

                    # Mass-weighted median diameter [mm]
                    if cumMass[0]>=0.5*cumMass[-1]:
                        dmm[i] = p3psd['bin_mid'][0]
                    else:
                        dmm[i] = p3psd[
                            'bin_mid'][np.where(cumMass>0.5*cumMass[-1])[0][0]-1]

    p3psd['sclwp'] = np.ma.masked_where(np.sum(error, axis=1)==0., rmass)
    p3psd['riming_frac'] = np.ma.masked_where(np.sum(error, axis=1)==0., rfrac)
    p3psd['a_coeff'] = np.ma.masked_where(np.sum(error, axis=1)==0., a_coeff)
    p3psd['b_coeff'] = np.ma.masked_where(np.sum(error, axis=1)==0., b_coeff)
    if compute_fits:
        p3psd['mu_ls'] = np.ma.masked_where(N0==0., mu)
        p3psd['lambda_ls'] = np.ma.masked_where(N0==0., lam)
        p3psd['N0_ls'] = np.ma.masked_where(N0==0., N0)
    p3psd['Nw_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., Nw)
    p3psd['iwc_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., iwc)
    p3psd['mean_aspect_ratio_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., asr)
    p3psd['dm_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dm)
    p3psd['dmm_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dmm)
    p3psd['eff_density_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., rho_eff)
    p3psd['dfr_KuKa_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dfr_KuKa)

    return p3psd

def calc_chisquare(
    x, Nt_obs, iwc_obs, z_obs, bin_mid, bin_width, a_coefficient, b_coefficient,
    rime_ind=None, exponential=False):
    '''
    Compute gamma fit parameters for the PSD.
    Follows McFarquhar et al. (2015) by finding N0-mu-lambda minimizing first
    (Nt), third (mass), sixth (reflectivity) moments.
    Inputs:
        x: N0, mu, lambda to test on the minimization procedure
        Nt_obs: Observed number concentration [L^-1]
        iwc_obs: Observed IWC using an assumed m-D relation [g m**-3]
        z_obs: Observed Z (following Hogan et al. 2012 definition) using assumed m-D relation [mm**6 m**-3]
        bin_mid: Midpoints for the binned particle size [mm]
        bin_width: Bin width for the binned particle size [cm]
        a_coefficient: Prefactor component to the assumed m-D reltation [cm**-b]
        b_coefficient: Exponent component to the assumed m-D reltation
        rime_ind (optional, for LS products only): Riming category index to use for the reflectivity moment
        exponential: Boolean, True if setting mu=0 for the fit (exponential form)
    Outputs:
        chi_square: Chi-square value for the provided N0-mu-lambda configuration
    '''
    Dmax = bin_mid / 10. # midpoint in cm
    dD = bin_width # bin width in cm
    mass_particle = a_coefficient * Dmax**b_coefficient # binned particle mass [g]

    if exponential: # exponential form with mu=0
        ND_fit = x[0] * np.exp(-x[2]*Dmax)
    else: # traditional gamma function with variable mu
        ND_fit = x[0] * Dmax**x[1] * np.exp(-x[2]*Dmax)
        
    Nt_fit = 1000.*np.nansum(ND_fit*dD) # L**-1
    iwc_fit = 10.**6  * np.nansum(mass_particle*ND_fit*dD) # g m**-3
    if rime_ind is not None:
        Z_fit = forward_Z() #initialize class
        Z_fit.set_PSD(PSD=ND_fit[np.newaxis,:]*10.**8, D=Dmax/100., dD=dD/100., Z_interp=True) # get the PSD in the format to use in the routine (mks units)
        Z_fit.load_split_L15() # Load the leinonen output
        Z_fit.fit_sigmas(Z_interp=True) # Fit the backscatter cross-sections
        Z_fit.calc_Z() # Calculate Z...outputs are Z.Z_x, Z.Z_ku, Z.Z_ka, Z.Z_w for the four radar wavelengths
        z_fit = 10.**(Z_fit.Z_x[0, rime_ind] / 10.) # mm**6 m**-3
    else:
        z_fit = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*ND_fit*dD) # mm**6 m**-3

    csq_Nt = ((Nt_obs-Nt_fit) / np.sqrt(Nt_obs*Nt_fit))**2
    csq_iwc = ((iwc_obs-iwc_fit) / np.sqrt(iwc_obs*iwc_fit))**2
    csq_z = ((z_obs-z_fit) / np.sqrt(z_obs*z_fit))**2
    chi_square = [csq_Nt, csq_iwc, csq_z]

    return chi_square