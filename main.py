from lib import DataLoader, default_collate, collate_fn


# def dem0_test_1():
#     from dataset import SimpleV1Dataset
#     simple_dataset = SimpleV1Dataset()  # dataset
#     dataloader = DataLoader(simple_dataset, batch_size=2, collate_fn=default_collate)
#     for data in dataloader:
#         print(data)

def coco_test():
    from dataset import CocoDataset
    coco_dataset = CocoDataset()

    # for i, val in enumerate(coco_dataset):
    #     print(val)
    #     print(i)

    # dataloader = DataLoader(coco_dataset, batch_size=2, collate_fn=default_collate)
    dataloader = DataLoader(coco_dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)


if __name__ == '__main__':
    # dem0_test_1()
    coco_test()
