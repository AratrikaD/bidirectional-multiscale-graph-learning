import geopandas as gpd
import libpysal as lp
import pandas as pd

file_path = "../../data/WijkBuurtkaart_2024_v1/wijkenbuurten_2024_v1.gpkg"
# C:\Users\AratrikaD\gnns-for-property-valuation\data\WijkBuurtkaart_2024_v1\wijkenbuurten_2024_v1.gpkg
layers = ["buurten", "wijken", "gemeenten"]
for layer in layers:
    gdf = gpd.read_file(filename=file_path, layer=layer)

    gdf_neighbors = lp.weights.Queen.from_dataframe(gdf, use_index=False)

    gdf_neighbors.to_sparse()

    codes = gdf.iloc[:, :1].to_numpy().flatten()
    adj_df = pd.DataFrame.sparse.from_spmatrix(
        gdf_neighbors.to_sparse(), index=codes, columns=codes
    )

    adj_df.info()