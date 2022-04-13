from hnmchallenge.datasets.last2month_last_day import L2MLDDataset
from hnmchallenge.datasets.last_month_last_day import LMLDDataset
from hnmchallenge.datasets.last_month_last_week_dataset import LMLWDataset
from hnmchallenge.datasets.last_month_last_week_user import LMLUWDataset
from hnmchallenge.datasets.last_week_last_week import LWLWDataset
from hnmchallenge.features.item_features import *
from hnmchallenge.features.light_gbm_features import *
from hnmchallenge.features.user_features import *
from hnmchallenge.features.user_item_features import *
from hnmchallenge.utils.logger import set_color
from tqdm import tqdm

USER_FEATURES = [
    UserAvgBuySession,
    Active,
    Age,
    ClubMemberStatus,
    FashionNewsFrequency,
    Fn,
    AvgPrice,
    UserTendency,
    UserTendencyLM,
    SaleChannelScore,
    UserAvgBuyDay,
]
GBM_FEATURES = [
    GraphicalAppearanceNOGBM,
    IndexCodeGBM,
    IndexGroupNameGBM,
    ProductGroupNameGBM,
]
ITEM_FEATURES = [
    ColourGroupCode,
    DepartmentNO,
    GarmentGroupName,
    GraphicalAppearanceNO,
    GarmentGroupNO,
    IndexCode,
    IndexGroupName,
    IndexGroupNO,
    ItemCount,
    ItemCountLastMonth,
    NumberBought,
    PerceivedColourMasterID,
    PerceivedColourValueID,
    ProductGroupName,
    ProductTypeNO,
    SectionNO,
    Price,
    SalesFactor,
    ItemSaleChannelScore,
    ItemAgePop,
    PopSales1,
    PopSales2,
    ItemPriceProduct,
]
USER_ITEM_FEATURES = [
    TimeScore,
    TimeWeight,
    TimesItemBought,
]

if __name__ == "__main__":
    # TODO RECHECK ALL THE FEATURES AND RECREATE THE ONE-HOT ONE
    dataset = LMLUWDataset()
    print(f"Creating features for DATASET:{dataset.DATASET_NAME}\n {print(dataset)}")

    # create the user features
    print("Saving User features")
    if len(USER_FEATURES) > 0:
        for uf in tqdm(USER_FEATURES):
            print(set_color(f"Saving {uf.FEATURE_NAME}...", "cyan"))
            for kind in ["train", "full"]:
                f = uf(kind=kind, dataset=dataset)
                f.save_feature()

    # create gbm features
    print("Saving gbm features")
    if len(GBM_FEATURES) > 0:
        for uf in tqdm(GBM_FEATURES):
            print(set_color(f"Saving {uf.FEATURE_NAME}...", "cyan"))
            for kind in ["train", "full"]:
                f = uf(kind=kind, dataset=dataset)
                f.save_feature()

    # create the item features
    print("Saving Item features")
    if len(ITEM_FEATURES) > 0:
        for uf in tqdm(ITEM_FEATURES):
            print(set_color(f"Saving {uf.FEATURE_NAME}...", "cyan"))
            for kind in ["train", "full"]:
                f = uf(kind=kind, dataset=dataset)
                f.save_feature()

    # create the context features
    print("Saving Context features")
    if len(USER_ITEM_FEATURES) > 0:
        for uf in tqdm(USER_ITEM_FEATURES):
            print(set_color(f"Saving {uf.FEATURE_NAME}...", "cyan"))
            for kind in ["train", "full"]:
                f = uf(kind=kind, dataset=dataset)
                f.save_feature()
