"""Create dataset from all csv files"""
import pandas as pd
from pandasql import sqldf

FILES: list[str] = ["additional_description.csv", "attributes.csv", "category.csv", "country.csv", "product_name.csv"] 

def get_merged_dataframe(file_path: str) -> pd.DataFrame:
    """Create dataframe from 5 csv files

    Args:
        file_path (str): Path to csv files

    Returns:
        pd.DataFrame: Final dataframe
    """
    add_desc = pd.read_csv(f"{file_path}/additional_description.csv").dropna(subset="id_product").astype({'id_product': 'int32'}).drop_duplicates()
    attribute = pd.read_csv(f"{file_path}/attributes.csv").dropna(subset="id_product").astype({'id_product': 'int32'}).drop_duplicates()
    category = pd.read_csv(f"{file_path}/category.csv").dropna(subset="product_name").drop_duplicates()
    country = pd.read_csv(f"{file_path}/country.csv").dropna(subset="id_product").astype({'id_product': 'int32'}).drop_duplicates()
    product_name = pd.read_csv(f"{file_path}/product_name.csv").dropna(subset="id_product").astype({'id_product': 'int32'}).drop_duplicates()
    pysqldf = lambda q: sqldf(q, globals())
    q = """select pn.id_product, pn.product_name, price, merchant_name, brand_name, product_description, category
        from product_name as pn
        inner join country as c on pn.id_product=c.id_product
        inner join attribute as a on pn.id_product=a.id_product
        inner join add_desc as ad on pn.id_product=ad.id_product
        inner join category cat on cat.product_name=pn.product_name
        where uk is not null
        """
    return pysqldf(q)


class CleanDataset:
    
    def __init__(self, price_transformation: str, categories_threshold: float) -> None:
        """_summary_

        Args:
            price_transformation:
            categories_threshold:
        """
        if price_transformation not in ["mean", "median"]:
            raise ValueError("price_transformation arg should be mean or median")
        self._price_transformation = price_transformation
        self._categories_threshold = categories_threshold
    
    def _handle_price(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute mean or median of transaction price
        
        Args:
            dataset: Dataset containing price

        Returns:
            pd.DataFrame: _description_
        """
        if self._price_transformation == "mean":
            price = dataset[["id_product", "price"]].groupby("id_product").mean().reset_index()
        else:
            price = dataset[["id_product", "price"]].groupby("id_product").median().reset_index()
        return dataset.drop(columns="price").drop_duplicates().merge(price, on="id_product")
    
    def _filter_categories(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Keep only categories representing more than categories_threshold pct.

        Args:
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        nb_product_category = dataset[["id_product", "category"]].groupby("category").count().reset_index().rename(columns={"id_product": "product_number"})
        nb_product_category['percentage'] = (nb_product_category.product_number / dataset.id_product.count())*100
        category_ro_remove = nb_product_category[nb_product_category.percentage < self._categories_threshold].category.tolist()
        return dataset[~dataset.category.isin(category_ro_remove)]
        
    def _multilabel_transformation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Join categories for each product
        
        Args:
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        merged_categories = dataset[["id_product", "category"]].groupby("id_product").category.apply(list).reset_index().rename(columns={"category": "categories"})
        return dataset.drop(columns="category").drop_duplicates().merge(merged_categories, on="id_product")
    
    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """_summary_
        
        Args:
            dataset: Input dataset to clean

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        # Remove duplicates
        dataset = dataset.drop_duplicates(subset=["product_name", "price", "merchant_name", "brand_name", "product_description", "category"])
        # Remove uncategorized class
        dataset = dataset[dataset.category != "uncategorized"]
        # Remove transactions
        dataset = self._handle_price(dataset=dataset)
        # Remove smaller classes
        dataset = self._filter_categories(dataset=dataset)
        # Join categories as list in order to obtain 1 line per product
        dataset = self._multilabel_transformation(dataset=dataset)
        
        return dataset
