#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

const int width = 1920;
const int height = 1080;
const int max_iter = 100;
const int r = 2;

const int powered_r = pow(r, 2);
Mat image;

int main(int argc, char* argv[])
{
    int thread, thread_size;
    bool done;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_size);

    if (thread_size < 2){
        cout << "Недосаточно потоков" << endl;
        return -1;
    }
    if (thread == 0){
        int* intervals = new int[4];
        MPI_Status status;
        image = Mat(height, width, CV_8UC3, Scalar(0, 0, 0));
        for (int ithread = 1; ithread < thread_size; ithread++){
            intervals[0] = (height + 1) / (thread_size - 1) * (ithread - 1);
            intervals[2] = (width + 1) / (thread_size - 1) * (ithread - 1);
            if (ithread == thread_size - 1){
                intervals[1] = height + 1;
                intervals[3] = width + 1;
            }else{
                intervals[1] = (height + 1) / (thread_size - 1) * ithread;
                intervals[3] = (width + 1) / (thread_size - 1) * ithread;
            }
            MPI_Send(&intervals, 4, MPI_INT, ithread, 0, MPI_COMM_WORLD);
            cout << "Отправили -> " << ithread << ": " << intervals[0] << ", " << intervals[1] << ", " << intervals[2] << ", " << intervals[3] << endl;
        }
        for (int ithread = 1; ithread < thread_size; ithread++){
            bool temp;
            MPI_Recv(&temp, 1, MPI_INT, ithread, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            cout << "Приняли от <- " << ithread << endl;
        }
        // namedWindow("Result", WINDOW_NORMAL);
        imshow("Result", image);
        waitKey(0);
        delete[] intervals;
        destroyAllWindows();
    } else {
        int* intrvls = new int[4];
        MPI_Status status;
        MPI_Recv(&intrvls, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        cout << "Поток " << thread << " получил интервалы: " << intrvls[0] << ", " << intrvls[1] << ", " << intrvls[2] << ", " << intrvls[3] << endl;
        for (int i = intrvls[0]; i < intrvls[1]; i++){
            for (int j = intrvls[2]; j < intrvls[3]; j++){
                double z_r = 0;
                double z_i = 0;
                double c_r = (double) j / width * 3.5 - 2.5;
                double c_i = (double) i / height * 2.0 - 1.0;
                bool flag = false;
                for (int iter = 0; iter < max_iter; iter++){
                    double z_new_r = pow(z_r, 2) - pow(z_i, 2) + c_r;
                    double z_new_i = 2 * z_r * z_i + c_i;
                    z_r = z_new_r;
                    z_i = z_new_i;
                    if (pow(z_r, 2) + pow(z_i, 2) > powered_r){
                        int n = (double)iter / (double)max_iter * 510;
                        if (n <= 255){
                            image.at<Vec3b>(i, j)[0] = 255;
                            image.at<Vec3b>(i, j)[1] = n;
                            image.at<Vec3b>(i, j)[2] = n;
                        }else{
                            n -= 255;
                            image.at<Vec3b>(i, j)[0] = 255 - n;
                            image.at<Vec3b>(i, j)[1] = 255 - n / 2;
                            image.at<Vec3b>(i, j)[2] = 255;
                        }
                        break;
                    }
                }
            }
        }
        done = true;
        MPI_Send(&done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        delete[] intrvls;
    }
    MPI_Finalize();
    return 0;
}
