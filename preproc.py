import mne






def GetRawData(subjects, run):
    
    fpath_subject = []
    fpath_run     = []
    fpath         = []
    k             = 0

    for i in subjects:
        print(i)
        fpath_subject_temp = "/home/groenborg/mne_data/MNE-upperlimb-data/record/834976/files/motorimagination_subject"+str(i)
        fpath_subject.append(fpath_subject_temp)
        print(fpath_subject,len(fpath_subject))

    for i in run:
        fpath_run_temp = "_run"+str(i)+".gdf"
        fpath_run.append(fpath_run_temp)
        print(fpath_run,len(fpath_run))

    for i in fpath_subject:
        print("im in a for loop")
        for k in fpath_run:
            print("im in while loop")
            fpath_temp = i+k
            fpath.append(fpath_temp)
            print(fpath)
    

    raw = mne.concatenate_raws([mne.io.read_raw_gdf(f, 
                        eog=['eog-l', 'eog-m', 'eog-r'],
                        preload=True, 
                        include= ("C5","C3","C1","Cz","C2", "C4", "C6" ,"P3","P1","Pz","P2","P4")
                        #exclude=('thumb_near', 'thumb_far', 'thumb_index', 'index_near', 'index_far', 'index_middle', 'middle_near', 'middle_far', 'middle_ring', 'ring_near', 'ring_far', 'ring_little', 'litte_near', 'litte_far', 'thumb_palm', 'wrist_bend', 'roll', 'pitch', 'gesture', 'handPosX', 'handPosY', 'handPosZ', 'elbowPosX', 'elbowPosY', 'elbowPosZ', 'ShoulderAdductio', 'ShoulderFlexionE', 'ShoulderRotation', 'Elbow', 'ProSupination', 'Wrist', 'GripPressure','armeodummy')
                        ) for f in fpath])
    
    return raw


subjects = [1]
run = [1,2,3,4,5]

raw = GetRawData(subjects,run)
raw.apply_function(lambda x: x * 1e-6)
raw.set_eeg_reference(ref_channels = 'average')

events,event_id = mne.events_from_annotations(raw)

event_dict = {
    "Elbow flex"  :   1,
    #"Elbow extend":   2
    #"Supination"  :   3,
    #"Pronatinon"  :   4,
    #"Hand close"  :   5,
    #"Hand open"   :   6,
    "Rest"        :   7
}

tmin = 1
tmax = 3

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    baseline=None,
    preload=True,
)
