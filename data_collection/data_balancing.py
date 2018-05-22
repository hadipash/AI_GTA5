import h5py

from data_collection.data_collect import path as source_path

dest_path = "F:\Graduation_Project\\training_data_balanced.h5"

destination = h5py.File(dest_path, 'w')
destination.create_dataset('img', (0, 240, 320, 3), dtype='u1', maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
destination.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
destination.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))


def save(data_img, controls, metrics):
    if data_img:  # if the list is not empty
        destination["img"].resize((destination["img"].shape[0] + len(data_img)), axis=0)
        destination["img"][-len(data_img):] = data_img
        destination["controls"].resize((destination["controls"].shape[0] + len(controls)), axis=0)
        destination["controls"][-len(controls):] = controls
        destination["metrics"].resize((destination["metrics"].shape[0] + len(metrics)), axis=0)
        destination["metrics"][-len(metrics):] = metrics


def main():
    source = h5py.File(source_path, 'r')
    images = []
    controls = []
    metrics = []

    tuples = 0
    straights = 0
    for i in range(source['img'].shape[0]):
        # if speed is not 0 and not arrived at the destination
        if source['metrics'][i][0] != 0 and source['metrics'][i][1] != 6:
            # save only each 5th straight drive frame
            if source['controls'][i][1] == 0:
                add = (straights % 5 == 0)
                straights += 1
            # save all turns
            else:
                add = True

            if add:
                images.append(source['img'][i])
                controls.append(source['controls'][i])
                metrics.append(source['metrics'][i])
                tuples += 1

                if tuples % 10000 == 0:  # every 2.5 GB
                    print(tuples)
                    save(images, controls, metrics)
                    images = []
                    controls = []
                    metrics = []

    save(images, controls, metrics)
    print("Copied: {:d} tuples from the source file".format(tuples))

    source.close()
    destination.close()


if __name__ == '__main__':
    main()
