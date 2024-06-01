import rasterio
import numpy as np

class Fmask():

    def read_landsat_bands(self, tif_file: str) -> np.ndarray:
        """Read the tif file

        Args:
            tif_file (str): A string with the path to the tif

        Returns:
            _type_: ndarray with all tif bands
        """
        with rasterio.open(tif_file) as src:
            bands = src.read()
        return bands

    def calculate_ndvi(self,
                       b4: np.ndarray,
                       b3: np.ndarray) -> np.ndarray:
        
        """Calculate the NDVI spectral indice by: nir-red/nir+red

        Args:
            b4 (np.ndarray): Nir band for landsat4-5 TM and landsat7 ETM
            b3 (np.ndarray): Red band for landsat4-5 TM and landsat7 ETM

        Returns:
            np.ndarray: A single band with the np.ndarray 
        """
        ndvi = (b4 - b3) / (b4 + b3)
        return ndvi

    def calculate_ndsi(self, b2: np.ndarray, b5: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b2 (np.ndarray): _description_
            b5 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        return (b2 - b5) / (b2 + b5)

    def calculate_brightness_temperature(self, band_thermal, k1, k2, to_celsius=False):
        radiance = band_thermal * 0.05518 + 1.2378  # Ajuste conforme necessário
        bt_kelvin = k2 / np.log((k1 / radiance) + 1)
        if to_celsius:
            bt_celsius = bt_kelvin - 273.15
            return bt_celsius
        else:
            return bt_kelvin

    def basic_test(self, b7, bt, ndvi, ndsi) -> np.ndarray:
        """_summary_

        Args:
            b7 (_type_): _description_
            bt (_type_): _description_
            ndvi (_type_): _description_
            ndsi (_type_): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 1
        # bt = calculate_brightness_temperature(b6, 607.76, 1260.56, to_celsius=True)
        result = np.logical_and(b7 > 0.03, bt < 27)
        result = np.logical_and(result, ndvi < 0.8)
        result = np.logical_and(result, ndsi < 0.8)

        return result

    def calculate_mean_visible(self,
                               b3: np.ndarray, 
                               b2: np.ndarray, 
                               b1: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b3 (np.ndarray): _description_
            b2 (np.ndarray): _description_
            b1 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        return (b3 + b2 + b1) / 3

    def whiteness_test(self, b3: np.ndarray, b2: np.ndarray, b1: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b3 (np.ndarray): _description_
            b2 (np.ndarray): _description_
            b1 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 2
        bands = np.array([b3, b2, b1])
        mean_visible = self.calculate_mean_visible(b3, b2, b1)
        whiteness = np.zeros_like(b3)

        for band in bands:
            # whiteness += np.abs(np.divide((np.subtract(band, mean_visible)), mean_visible))
            whiteness += np.abs((band - mean_visible)/mean_visible)

        return whiteness < 0.7

    def hot_test(self, b1: np.ndarray, b3: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b1 (np.ndarray): _description_
            b3 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 3
        # return np.multiply(np.subtract(b1, 0.5), np.subtract(b3, 0.08)) > 0
        b3 = 0.5 * b3
        hot_test = b1 - b3 - 0.08
        hot_test = hot_test > 0
        return hot_test

    def b4_over_b5_test(self, b4: np.ndarray, b5: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            b4 (np.ndarray): _description_
            b5 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 4
        return np.divide(b4, b5) > 0.75

    def pass_one(self,
                 b1: np.ndarray,
                 b3: np.ndarray,
                 b4: np.ndarray,
                 b5: np.ndarray,
                 b7: np.ndarray,
                 bt: np.ndarray,
                 ndvi: np.ndarray,
                 ndsi: np.ndarray,
                 whiteness: np.ndarray):
        
        # Eq. 6
        pcp = np.logical_and(self.basic_test(b7, bt, ndvi, ndsi), whiteness) 
        pcp = np.logical_and(pcp, self.hot_test(b1, b3))
        pcp = np.logical_and(pcp, self.b4_over_b5_test(b4, b5))

        return pcp

    def water_test(self, ndvi: np.ndarray, b4: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            ndvi (np.ndarray): _description_
            b4 (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        
        # Eq. 5
        return np.logical_or(np.logical_and(ndvi < 0.01, b4 < 0.11),
                            np.logical_and(ndvi < 0.1, b4 < 0.05))

    def clear_sky_water_test(self, water_test: np.array, b7: np.ndarray) -> np.ndarray:
        return np.logical_and(water_test, b7 < 0.03)

    def water_cloud_prob(self, 
                         water_test: np.ndarray,
                         b5: np.ndarray, 
                         b7: np.ndarray,
                         bt: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            water_test (np.ndarray): _description_
            b5 (np.ndarray): _description_
            b7 (np.ndarray): _description_
            bt (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        # Eq. 7
        clear_sky_water = self.clear_sky_water_test(water_test=water_test, b7=b7)

        # Eq. 8
        t_water = np.percentile(bt[clear_sky_water], 82.5)

        # Eq. 9
        w_temperature_prob = (t_water - bt) / 4

        # Eq. 10
        print(b5)
        brightness_prob = np.minimum(b5, 0.11) / 0.11

        # Eq. 11
        w_cloud_prob = w_temperature_prob * brightness_prob

        return w_cloud_prob
    
    def land_cloud_prob(self, pcp: np.ndarray, water_test: np.ndarray, bt: np.ndarray, whiteness: np.ndarray):
        # Eq. 12
        clear_sky_land = np.logical_not(pcp) & np.logical_not(water_test)

        # Eq. 13
        t_low = np.percentile(bt[clear_sky_land], 17.5)
        t_high = np.percentile(bt[clear_sky_land], 82.5)

        # Eq. 14
        l_temperature_prob = (t_high + 4 - bt) / (t_high + 4 - (t_low - 4))
        
        # TODO Create a function to modificate the NDSI and NDVI indices 
        modified_ndvi = np.zeros_like(pcp)
        modified_ndsi = np.zeros_like(pcp)

        # Eq. 15
        print(type(modified_ndsi), type(modified_ndvi), type(whiteness))
        maximum = np.maximum(np.abs(modified_ndvi), np.abs(modified_ndvi))
        variability_prob = 1 - np.maximum(maximum, whiteness)

        return None

    def pass_two():
        pass


    def detect_clouds(self, bands: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            bands (np.ndarray): _description_
            tif_file (str): _description_

        Returns:
            np.ndarray: _description_
        """

        B1 = bands[0]
        B2 = bands[1]
        B3 = bands[2]
        B4 = bands[3]
        B5 = bands[4]
        B6 = bands[5]
        B7 = bands[6]  

        ndvi = self.calculate_ndvi(B4, B3)
        ndsi = self.calculate_ndsi(B2, B5)
        bt = B6 - 273.15
        whiteness = self.whiteness_test(B3, B2, B1)

        # Pass One        
        pcp = self.pass_one(b1=B1, b3=B3, b4=B4, b5=B5, b7=B7, bt=bt,
                            ndvi=ndvi,
                            ndsi=ndsi,
                            whiteness=whiteness)

        # Pass Two
        water = self.water_test(ndvi, B7)        

        w_cloud_prob = self.water_cloud_prob(water_test=water,
                                             b5=B5,
                                             b7=B7,
                                             bt=bt)


        l_cloud_prob = self.land_cloud_prob(pcp=pcp,
                                            water_test=water,
                                            bt=bt,
                                            whiteness=whiteness)

        return pcp

    def detect_shadows(self, bands, cloud_mask):
        swir2 = bands[5]
        thermal = bands[6]

        shadow_mask = (thermal < 290) & cloud_mask
        
        return shadow_mask
    
    def save(self, band: np.ndarray, tif_file: str, output_file: str) -> None:
        """_summary_

        Args:
            band (np.ndarray): _description_
            tif_file (str): _description_
            output_file (str): _description_
        """
        
        with rasterio.open(tif_file) as src:
            profile = src.profile
            profile.update(count=1)
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(band, 1)

    def create_fmask(self, tif_file: str) -> np.ndarray:
        """_summary_

        Args:
            tif_file (str): _description_

        Returns:
            np.ndarray: _description_
        """

        bands = self.read_landsat_bands(tif_file)
        cloud_mask = self.detect_clouds(bands)
        # shadow_mask = self.detect_shadows(bands, cloud_mask)

        fmask = np.zeros_like(cloud_mask, dtype=np.uint8)
        # fmask = np.ones_like(cloud_mask, dtype=np.uint8)
        fmask[cloud_mask] = 1
        # fmask[shadow_mask] = 2

        print(fmask)
        
        return cloud_mask


if __name__ == "__main__":
    # input_tif = "./seixas_20110808.tif"
    input_tif = "./seixas_20110128.tif"
    output_tif = "./outputs_toa.tif"
    fmask = Fmask()
    result = fmask.create_fmask(input_tif)
    fmask.save(result, input_tif, output_tif)
