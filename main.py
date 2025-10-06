import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import date

def define_shark_range(
        chloro_img: str,
        ptk_img: str,
        temp_img: str,
        chlro_threshold: int,
        ptk_threshold: int,
        temp_threshold: tuple,
        lat_ranges: list = [(-30, 30)],
        lon_ranges: list = [(-180, 180)]
) -> np.ndarray:
    # Load grayscale images
    chloro = cv2.imread(chloro_img, cv2.IMREAD_GRAYSCALE)
    ptk = cv2.imread(ptk_img, cv2.IMREAD_GRAYSCALE)
    temp = cv2.imread(temp_img, cv2.IMREAD_GRAYSCALE)

    # Apply thresholds
    chloro_mask = chloro >= chlro_threshold
    ptk_mask = ptk >= ptk_threshold
    temp_mask = (temp >= temp_threshold[0]) & (temp <= temp_threshold[1])

    # Overlapping region
    overlap_mask = chloro_mask & ptk_mask & temp_mask

    # Create latitude mask
    h, w = overlap_mask.shape
    lat_mask = np.zeros_like(overlap_mask, dtype=bool)
    for lat_min, lat_max in lat_ranges:
        y_min = int((90 - lat_max) / 180 * h)
        y_max = int((90 - lat_min) / 180 * h)
        lat_mask[y_min:y_max, :] = True

    # Create longitude mask
    lon_mask = np.zeros_like(overlap_mask, dtype=bool)
    for lon_min, lon_max in lon_ranges:
        x_min = int((lon_min + 180) / 360 * w)
        x_max = int((lon_max + 180) / 360 * w)
        lon_mask[:, x_min:x_max] = True

    # Apply latitude & longitude restriction
    overlap_mask = overlap_mask & lat_mask & lon_mask

    return (overlap_mask.astype(np.uint8)) * 255

st.title("Shark habitat range based on chlorophyll, phytoplankton, and sea temperature")
st.divider()

shark_species_list = ['Great White', 'Greenland Shark', 'Basking Shark']
shark_species_info = {
    'Great White': ((150, 255), [(-60, 60)], [(-180, 180)]),       # global range
    'Greenland Shark': ((0, 170), [(48, 83)], [(-75, 45)]),        # N Atlantic/Arctic
    'Basking Shark': ((0, 150), [(-40, 60)], [(-180, 180)])        # mostly global temperate waters
}

shark_species = st.selectbox('Select a shark species', shark_species_list)
selected_date = st.date_input("Select a date", date.today())

if shark_species:
    temp_threshold, lat_ranges, lon_ranges = shark_species_info[shark_species]
    img = define_shark_range(
        'images/clorofila.png',
        'images/fitoplancton.png',
        'images/temperatura.png',
        150,
        150,
        temp_threshold,
        lat_ranges,
        lon_ranges
    )

    st.image(img, caption=f'{shark_species} habitat range', use_container_width=True)

    ys, xs = np.where(img == 255)
    h, w = img.shape


    lons = (xs / w) * 360 - 180
    lats = 90 - (ys / h) * 180


    df = pd.DataFrame({'lat': lats, 'lon': lons})

    if len(df) > 5000:
        df = df.sample(5000)


    st.map(df, color="#FF0000", zoom=1)
