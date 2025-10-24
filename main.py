from src.data.init_dataset import InitDataset
# from src.preprocessing.data_filtering import filter_master_data

def main():
    # dataset 초기화 및 통합
    init_dataset = InitDataset()
    master_data = init_dataset.integrate_data()

    # filter_master_data(master_data)


if __name__ == "__main__":
    main()


## 12
