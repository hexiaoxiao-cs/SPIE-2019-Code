import datetime
import glob
import os
import pickle
import random
import shutil
import socket
import SimpleITK as sitk
import numpy as np
import tqdm
from Training import config_for_automation as trainer
from unet3d.prediction import run_validation_cases
import sys
##For multiprocessing

import threading


def createoutputs(prediction_dir):
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


def createtrainingdata(i, rootdir, k,numcontrolpoints=5, stdDef=15):
    # Create STD deformation Map

    # Last TrainingData+Distort
    currdir = os.path.join(rootdir, "0-time", "data", "1")
    datato = os.path.join(rootdir, str(i) + "-time", "data"+str(k)+str(config["initial_learning_rate"])) ##folder for k set
    print("Last TrainingData+Distort")
    if os.path.exists(os.path.join(datato, "1")):
        return
    for subject_folder in tqdm.tqdm(glob.glob(os.path.join(currdir, "*"))):
        # print(subject_folder)
        transfromDomainMeshSize = [numcontrolpoints] * 3
        paramsNp = np.zeros(1536)
        paramsNp = paramsNp + np.random.randn(1536) * stdDef
        paramsNp[0:int(1536 / 3)] = 0  # REMOVE Z DEFORMATIONS
        params = tuple(paramsNp)
        if os.path.isdir(subject_folder):
            ##readAugmented_0,2,3+truth
            Augmented_0 = sitk.ReadImage(os.path.join(subject_folder, "Augmented_0.nii.gz"))
            Augmented_2 = sitk.ReadImage(os.path.join(subject_folder, "Augmented_2.nii.gz"))
            Augmented_4 = sitk.ReadImage(os.path.join(subject_folder, "Augmented_4.nii.gz"))
            truth = sitk.ReadImage(os.path.join(subject_folder, "truth.nii.gz"))
            tx = sitk.BSplineTransformInitializer(Augmented_0, transfromDomainMeshSize)
            tx.SetParameters(params)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(Augmented_0)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(tx)
            out_Augmented_0 = resampler.Execute(Augmented_0)
            out_Augmented_2 = resampler.Execute(Augmented_2)
            out_Augmented_4 = resampler.Execute(Augmented_4)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            out_truth = resampler.Execute(truth)
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(datato, "1",
                                              subject)
            os.makedirs(new_subject_folder)
            sitk.WriteImage(out_Augmented_0, os.path.join(new_subject_folder, "Augmented_0.nii.gz"))
            sitk.WriteImage(out_Augmented_2, os.path.join(new_subject_folder, "Augmented_2.nii.gz"))
            sitk.WriteImage(out_Augmented_4, os.path.join(new_subject_folder, "Augmented_4.nii.gz"))
            sitk.WriteImage(out_truth, os.path.join(new_subject_folder, "truth.nii.gz"))
    ##Last PredictionData+Distort
    currdir = os.path.join(rootdir, str(i - 1) + "-time", "prediction_"+str(k)+"_"+str(config["initial_learning_rate"]))
    print("Last PredictionData+Distort")
    for subject_folder in tqdm.tqdm(glob.glob(os.path.join(currdir, "*"))):
        transfromDomainMeshSize = [numcontrolpoints] * 3
        paramsNp = np.zeros(1536)
        paramsNp = paramsNp + np.random.randn(1536) * stdDef
        paramsNp[0:int(1536 / 3)] = 0  # REMOVE Z DEFORMATIONS
        params = tuple(paramsNp)
        if os.path.isdir(subject_folder):
            ##readAugmented_0,2,3+truth
            Augmented_0 = sitk.ReadImage(os.path.join(subject_folder, "data_Augmented_0.nii.gz"))
            Augmented_2 = sitk.ReadImage(os.path.join(subject_folder, "data_Augmented_2.nii.gz"))
            Augmented_4 = sitk.ReadImage(os.path.join(subject_folder, "data_Augmented_4.nii.gz"))
            truth = sitk.ReadImage(os.path.join(subject_folder, "truth.nii.gz"))
            tx = sitk.BSplineTransformInitializer(Augmented_0, transfromDomainMeshSize)
            tx.SetParameters(params)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(Augmented_0)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(tx)
            out_Augmented_0 = resampler.Execute(Augmented_0)
            out_Augmented_2 = resampler.Execute(Augmented_2)
            out_Augmented_4 = resampler.Execute(Augmented_4)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            out_truth = resampler.Execute(truth)
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(datato, "2",
                                              subject)
            os.makedirs(new_subject_folder)
            sitk.WriteImage(out_Augmented_0, os.path.join(new_subject_folder, "Augmented_0.nii.gz"))
            sitk.WriteImage(out_Augmented_2, os.path.join(new_subject_folder, "Augmented_2.nii.gz"))
            sitk.WriteImage(out_Augmented_4, os.path.join(new_subject_folder, "Augmented_4.nii.gz"))
            sitk.WriteImage(out_truth, os.path.join(new_subject_folder, "truth.nii.gz"))
    # Copy Last val data
    if i>=2:
        shutil.copytree("../"+str(i-1)+"-time/data"+str(k)+str(config["initial_learning_rate"])+"/2",datato+"/3")

def traininger(addr1,addr2,out,i,k):
    config["training_file"] = addr1
    config["validation_file"] = addr2
    if k >= 1:
        with open(config["training_file"], "rb") as openpkl:
            kk = pickle.load(openpkl)
            numkk=len(kk)
        with open(config["validation_file"],"rb") as openpkl:
            vl=pickle.load(openpkl)
            num=len(vl)
        for q in range(50+num*(k-1), 50+num*k):
            kk.append(q)
        kk = kk[:numkk+num*k]
        # for i in range(6):
        #     rd = random.randint(0, 67)
        #     kk.append(kk[rd])
        with open(config["training_file"], "wb") as openpkl:
            pickle.dump(kk, openpkl)
        config["data_place"] = os.path.join("./data" + str(i) + str(config["initial_learning_rate"]))
        config["data_file"] = os.path.abspath("./brats_data"+str(i)+"_"+str(config["initial_learning"
                                                                                   "_rate"])+".h5")
    sys.stdout=open(out,"w")

    config["model_file"] = os.path.abspath("isensee_2017_model" + "_" + str(config["initial_learning_rate"]) +"_"+str(i)+ ".h5")

    print(config)
    trainer.config = config
    trainer.main()
    createoutputs(r"./prediction_"+str(i)+"_" + str(config["initial_learning_rate"]))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = trainer.config
date = datetime.date.today()
rootdir = r"./" + "2019-01-07" +"_"+socket.gethostname()
if not os.path.exists(rootdir):
    os.makedirs(rootdir)
rootdir = os.path.abspath(rootdir)
datafrom = r"../Augmenting/Augmented"
datafrom = os.path.abspath(datafrom)
os.chdir(rootdir)
mlist=list(range(50))
random.shuffle(mlist)
list_1=mlist[0:10]
list_2=mlist[10:20]
list_3=mlist[20:30]
list_4=mlist[30:40]
list_5=mlist[40:50]
training=list()
validation=list()
training.append(list_1+list_2+list_3+list_4)
validation.append(list_5)
training.append(list_1+list_2+list_3+list_5)
validation.append(list_4)
training.append(list_1+list_2+list_4+list_5)
validation.append(list_3)
training.append(list_1+list_3+list_4+list_5)
validation.append(list_2)
training.append(list_2+list_3+list_4+list_5)
validation.append(list_1)
trainingadr=list()
validationadr=list()
for i in range(0,5):
    with open(r"../training_"+str(i)+".pkl","wb") as openpkl:
        pickle.dump(training[i],openpkl)
        print(os.path.abspath(r"../training_"+str(i)+".pkl"))
        trainingadr.append(os.path.abspath(r"../training_"+str(i)+".pkl"))
    with open(r"../validation_" + str(i)+".pkl", "wb") as openpkl:
        pickle.dump(validation[i], openpkl)
        print(os.path.abspath(r"../validation_"+str(i)+".pkl"))
        validationadr.append(os.path.abspath(r"../validation_"+str(i)+".pkl"))


for i in range(0, 3):
    print("CurrIterOuter:" + str(i))
    nowdir = os.path.join(rootdir, str(i) + "-time")
    if not os.path.exists(nowdir):
        os.makedirs(nowdir)
    datato = os.path.join(nowdir, "data")
    if not os.path.exists(datato):
        os.makedirs(datato)
    if (i == 0) and not (os.path.exists(os.path.join(datato, "1"))):
        for subject_folder in tqdm.tqdm(glob.glob(os.path.join(datafrom, "*", "*"))):
            if os.path.isdir(subject_folder):
                subject = os.path.basename(subject_folder)
                new_subject_folder = os.path.join(datato, "1",
                                                  subject)
                # if not os.path.exists(new_subject_folder):
                # os.makedirs(new_subject_folder)
                shutil.copytree(subject_folder, new_subject_folder)

        ##Copy Old Model
        #
        # for k in range(0,4):
        #     currdir = os.path.join(rootdir, str(i - 1) + "-time",
        #                            "isensee_2017_model" + "_" + str(config["initial_learning_rate"]) + "_" + str(
        #                                k) + ".h5")
        #     shutil.copyfile(currdir,os.path.join(nowdir,"isensee_2017_model"+"_"+str(config["initial_learning_rate"])+"_"+str(k)+".h5"))
    ##ChangeCurrDir
    os.chdir(nowdir)
    config["data_file"] = os.path.abspath("brats_data.h5")

    # if i >= 1:
    #     with open(config["training_file"], "rb") as openpkl:
    #         kk = pickle.load(openpkl)
    #     for q in range(67, 77):
    #         kk.append(q)
    #     kk = kk[:77]
    #     # for i in range(6):
    #     #     rd = random.randint(0, 67)
    #     #     kk.append(kk[rd])
    #     with open(config["training_file"], "wb") as openpkl:
    #         pickle.dump(kk, openpkl)
    logout=list()
    for z in range(0,5):
        logout.append(os.path.join(os.path.abspath("./"),"training_sub_"+str(z)+".txt"))

    trainer.config = config
    print(config)
    print("startTraining")
    for k in range(0,5):
        if not i == 0:
            createtrainingdata(i, rootdir, k)
        print("Stage "+ str(k))
        #if i<1:
        #    continue

        traininger(trainingadr[k],validationadr[k],logout[k],k,i)
