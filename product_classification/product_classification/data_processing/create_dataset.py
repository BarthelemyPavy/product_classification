"""Create dataset from all csv files"""
import pandas as pd
from pandasql import sqldf
from product_classification import logger

FILES: list[str] = ["additional_description.csv", "attributes.csv", "category.csv", "country.csv", "product_name.csv"]


def get_merged_dataframe(file_path: str) -> pd.DataFrame:
    """Create dataframe from 5 csv files

    Args:
        file_path (str): Path to csv files

    Returns:
        pd.DataFrame: Final dataframe
    """
    add_desc = (
        pd.read_csv(f"{file_path}/additional_description.csv")
        .dropna(subset="id_product")
        .astype({'id_product': 'int32'})
        .drop_duplicates()
    )
    attribute = (
        pd.read_csv(f"{file_path}/attributes.csv")
        .dropna(subset="id_product")
        .astype({'id_product': 'int32'})
        .drop_duplicates()
    )
    category = pd.read_csv(f"{file_path}/category.csv").dropna(subset="product_name").drop_duplicates()
    country = (
        pd.read_csv(f"{file_path}/country.csv")
        .dropna(subset="id_product")
        .astype({'id_product': 'int32'})
        .drop_duplicates()
    )
    product_name = (
        pd.read_csv(f"{file_path}/product_name.csv")
        .dropna(subset="id_product")
        .astype({'id_product': 'int32'})
        .drop_duplicates()
    )
    logger.info("Create dataset")
    q = """select pn.id_product, pn.product_name, price, merchant_name, brand_name, product_description, category
        from product_name as pn
        inner join country as c on pn.id_product=c.id_product
        inner join attribute as a on pn.id_product=a.id_product
        inner join add_desc as ad on pn.id_product=ad.id_product
        inner join category cat on cat.product_name=pn.product_name
        where uk is not null
        """
    df = sqldf(q, locals())
    logger.info("Dataframe created")
    logger.info(f"Dataset shape: {df.shape}")
    return df


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning indentified on EDA

    Args:
        dataset: Input dataset to clean

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Remove duplicates
    dataset = dataset.drop_duplicates(
        subset=["product_name", "price", "merchant_name", "brand_name", "product_description", "category"]
    )
    logger.info("Duplicates removed")
    logger.info(f"Dataset shape: {dataset.shape}")
    # Change categorical columns to category type
    dataset["merchant_name"] = dataset["merchant_name"].astype("category")
    dataset["brand_name"] = dataset["brand_name"].astype("category")
    dataset["category"] = dataset["category"].astype("category")
    # Remove uncategorized class
    dataset = dataset[dataset.category != "uncategorized"]
    logger.info("Uncategorized class removed")
    logger.info(f"Dataset shape: {dataset.shape}")
    # Remove transactions
    dataset = dataset.drop_duplicates(subset=["id_product", "category"])
    logger.info("Remove duplicated pair (id_product, category)")
    logger.info(f"Dataset shape: {dataset.shape}")

    return dataset
