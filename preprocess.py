import os
import shutil

train_filenames = os.listdir('train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

rmrf_mkdir('train2')
os.mkdir('train2/cat')
os.mkdir('train2/dog')

test_filenames = os.listdir('test')
rmrf_mkdir('test2')
os.mkdir('test2/test')
for filename in test_filenames:
    shutil.copyfile('./test/'+filename, './test2/test/'+filename)

for filename in train_cat:
    shutil.copyfile('./train/'+filename, './train2/cat/'+filename)
    #os.symlink('../../train/'+filename, 'train2/cat/'+filename)

for filename in train_dog:
    shutil.copyfile('./train/' + filename, './train2/dog/' + filename)
    #os.symlink('../../train/'+filename, 'train2/dog/'+filename)


