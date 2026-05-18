# Geotechnical Stereonet

A Streamlit app for plotting slope and discontinuity set orientations on a lower-hemisphere stereonet, with screening-level planar sliding, wedge sliding, flexural toppling, and block toppling checks.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy On Streamlit Community Cloud

1. Go to <https://share.streamlit.io>.
2. Sign in with GitHub.
3. Create a new app from this repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Input Convention

- Dip direction is azimuth clockwise from north.
- Dip is in degrees from horizontal and must be between 0 and 90.
- Enter all discontinuities in one `Discontinuity Sets` table.
- The app uses lower-hemisphere stereographic projection.

## Kinematic Analysis

This is a screening-level kinematic analysis only. It checks orientation feasibility on a stereonet; it does not calculate factor of safety, block volume, persistence, groundwater pressure, cohesion, seismic loading, or release-plane geometry.

Planar sliding checks whether a discontinuity dips out of the slope, exceeds the friction angle, and daylights through the slope face.

Wedge sliding checks every pair of discontinuities and tests whether the intersection line plunges out of the slope, exceeds the friction angle, and daylights through the slope face.

Flexural toppling checks for steep discontinuities dipping into the slope using the threshold `dip > 90 - slope dip + friction angle`.

Block toppling is treated as a direct/block toppling screen. It checks discontinuity-pair intersections that plunge into the slope against the slope-angle, friction-angle, and lateral-limit envelope. It then checks whether a separate discontinuity pole plots in a valid direct-toppling release/base-plane zone, rather than accepting a base plane by dip alone. This is an orientation feasibility screen only; it is not a Goodman-Bray force equilibrium or block geometry model.

## Sources

- Rocscience Dips, Kinematic Analysis Overview: <https://www.rocscience.com/help/dips/v9/documentation/stereonet-2d/kinematic-analysis/kinematic-analysis-overview>
- Rocscience Dips, Planar Sliding: <https://www.rocscience.com/help/dips/v9/documentation/stereonet-2d/kinematic-analysis/planar-sliding>
- Rocscience Dips, Wedge Sliding: <https://www.rocscience.com/help/dips/v9/documentation/stereonet-2d/kinematic-analysis/wedge-sliding>
- Rocscience Dips, Flexural Toppling: <https://www.rocscience.com/help/dips/v9/documentation/stereonet-2d/kinematic-analysis/flexural-toppling>
- Rocscience Dips, Direct Toppling: <https://www.rocscience.com/help/dips/v9/documentation/stereonet-2d/kinematic-analysis/direct-toppling>
- Markland, J.T. (1972), `A useful technique for estimating the stability of rock slopes when the rigid wedge slide type of failure is expected`.
- Wyllie, D.C. and Mah, C.W., `Rock Slope Engineering: Civil Applications`.
