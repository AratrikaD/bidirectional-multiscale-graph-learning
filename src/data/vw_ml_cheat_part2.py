import geopandas as gpd
import libpysal as lp
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq


file_path_gebied = r"C:\Users\AratrikaD\gnns-for-property-valuation\housing-data\cbsgebiedsindelingen2016-2025\cbsgebiedsindelingen2023.gpkg"
# file_path_pc = r"C:\Users\juliane\git\itax\Datastore\ETL-Tools\Python-ETL\data\cbs_pc4_2023_v1.gpkg"

def find_regions(df, gdf, region_col="gemeentecode"):
    """
    Vectorized function to find regions for multiple points
    
    Parameters:
    df (DataFrame): DataFrame containing 'lat' and 'lon' columns
    gdf (GeoDataFrame): GeoDataFrame containing geometries to check against
    
    Returns:
    Series: Series of region codes
    """
    # Create GeoDataFrame from the points
    points_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT), crs="EPSG:4326")
    
    # Convert the CRS of points to match the GeoDataFrame
    points_gdf = points_gdf.to_crs(gdf.crs)
    
    # Perform spatial join
    joined_gdf = gpd.sjoin(points_gdf, gdf, how="left", predicate="within")
    
    return joined_gdf[region_col]

def read_file(file_path, layer):
    gdf = gpd.read_file(file_path, layer=layer)
    #gdf = gdf[gdf['indelingswijziging_wijken_en_buurten'] > 0] -- Wel of niet???
    gdf_neighbors = lp.weights.Queen.from_dataframe(gdf, use_index=False)
    gdf_neighbors.to_sparse()
    codes = gdf.iloc[:, :1].to_numpy().flatten()
    adj_df = pd.DataFrame.sparse.from_spmatrix(
        gdf_neighbors.to_sparse(), index=codes, columns=codes
    )
    return adj_df, gdf


def main():
    df_train = pd.read_csv(r"C:\Users\AratrikaD\gnns-for-property-valuation\housing-data\transaction_data.csv")
    # df_test = pd.read_csv(r"C:\Users\AratrikaD\gnns-for-property-valuation\housing-data\rotterdam_transaction_data.csv")

    
    layers = ["coropplusgebied_gegeneraliseerd", "gemeente_gegeneraliseerd", "buurt_gegeneraliseerd"]
    for layer in layers:
        _, gdf = read_file(file_path_gebied, layer)
        if layer == "gemeente_gegeneraliseerd":
            df_train['GEMEENTECODE'] = find_regions(df_train, gdf, region_col="statcode") 
            # df_test['GEMEENTECODE'] = find_regions(df_test, gdf, region_col="statcode") 
        elif layer == "wijk_gegeneraliseerd":
            df_train['CBS_DISTRICT'] = find_regions(df_train, gdf, region_col="statcode") 
            # df_test['CBS_DISTRICT'] = find_regions(df_test, gdf, region_col="statcode") 
        elif layer == "buurt_gegeneraliseerd":
            df_train['BUURTCODE'] = find_regions(df_train, gdf, region_col="statcode") 
            # df_test['BUURTCODE'] = find_regions(df_test, gdf, region_col="statcode")  
        elif layer == "coropplusgebied_gegeneraliseerd":
            df_train['COROPPLUSCODE'] = find_regions(df_train, gdf, region_col="statcode") 
            # df_test['COROPPLUSCODE'] = find_regions(df_test, gdf, region_col="statcode") 
        elif layer == "provincie_gegeneraliseerd":
            df_train['PROVINCECODE'] = find_regions(df_train, gdf, region_col="statcode") 
            # df_test['PROVINCECODE'] = find_regions(df_test, gdf, region_col="statcode") 

    # _, gdf = read_file(file_path_pc, layer=None)
    # df_train['POSTCODE'] = find_regions(df_train, gdf, region_col="postcode").astype('Int32') 
    # df_test['POSTCODE'] = find_regions(df_test, gdf, region_col="postcode").astype('Int32') 

    # lst_regions = ['MUNICIPALITYCODE', 'CBS_DISTRICT', 'CBS_NEIGHBORHOOD', 'PROVINCECODE', 'COROPPLUSCODE', 'ZIPCODE_NUMERIC']

    df_train = df_train.dropna(subset=["BUURTCODE","GEMEENTECODE", "COROPPLUSCODE"])
    # df_test = df_test.dropna(subset=["BUURTCODE","GEMEENTECODE", "COROPPLUSCODE"])

    df_train.to_csv(r"C:\Users\AratrikaD\gnns-for-property-valuation\housing-data\transaction_data.csv", index=False)
    # df_test.to_csv(r"C:\Users\AratrikaD\gnns-for-property-valuation\housing-data\rotterdam_transaction_data.csv", index=False)



if __name__ == "__main__":
    main()
