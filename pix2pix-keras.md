
## Keras implementation of https://phillipi.github.io/pix2pix

---

设置后端和有关环境变量，可以跳过，使用 tensorflow 和默认设置

```python
import os
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_compile,dnn.library_path=/usr/lib'
```

---

设置图片处理中 `channels_first` or `channels_last` 属性，由于使用 tf 后端，有关之处可以直接使用 `channels_last`

`channel_axis` 在后面 `BatchNormalization` 以及 `Concatenate` 层的 `axis` 参数中被用到，分别指定的是在哪个维度上连接或进行标准化。
这里猜测 `axis=-1` 的含义和 Python 切片中 `-1` 代表最后一个元素相同，即操作在最后一个维度进行

```python
import keras.backend as K
if os.environ['KERAS_BACKEND'] =='theano':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False
```

---

```python
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
```

---

> `__conv_init` 函数的使用是什么意思？尤其不知道 `RadomNormal(0, 0.02)` 之后进一步 call 了一个 `(a)`, 以及 `conv_weight` 这两个属性，keras documentation 里都没找到

`RandomNormal` 是 keras 官方的 `Initializer`, 见[文档](https://keras.io/initializers/), 第一个参数是 `mean`, 第二个参数是 `std_dev`，用于对 keras 层的初始权重进行随机初始化

```python
# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization
```

---

Theano 有关的设置，用 tf 的话不用管

```python
# HACK speed up theano
if K._BACKEND == 'theano':
    import keras.backend.theano_backend as theano_backend
    def _preprocess_conv2d_kernel(kernel, data_format):
        #return kernel
        if hasattr(kernel, "original"):
            print("use original")
            return kernel.original
        elif hasattr(kernel, '_keras_shape'):
            s = kernel._keras_shape
            print("use reshape",s)
            kernel = kernel.reshape((s[3], s[2],s[0], s[1]))
        else:
            kernel = kernel.dimshuffle((3, 2, 0, 1))
        return kernel
    theano_backend._preprocess_conv2d_kernel = _preprocess_conv2d_kernel
```

---

`conv2d`: 用默认的 `RandomNormal(0, 0.02)`, 也即上面定义的 `conv_init` 创建一个卷积层。
这里的 `f` 指的是 `filters` 数量，另一个 `Conv2D` 中必须的参数 `kernel_size` 需要在 `**k` 参数中指定

`batchnorm`: 一个神秘莫测的高深技术，具体参考[文档](https://keras.io/layers/normalization/)和[大佬写的博客](https://www.cnblogs.com/guoyaohua/p/8724433.html)

注意这里用到了 `channal_axis` 这个变量，在前面被初始化为了 `-1` （使用 tf）

```python
# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
```

---

### Discriminator

[CNN 基础概念理解](https://zhuanlan.zhihu.com/p/42559190)

> 设定输入层这里有一些问题：
> 1. shape 中包含 `None`, 官方文档中没有指出 Input 层可以为 `None` 的条件，根据[这个 issue ](https://github.com/keras-team/keras/issues/2054) 好像只有 Theano 的 RNN 可以这么做，具体等实操
> 2. `input_a` 和 `input_b` 分别来自哪里？看后面代码应该能解决

`nc_in` 和 `nc_out` 似乎指代的都是颜色的 Channel, 那么应该都是一样的，例如 `3`. 看后面 Generator 就直接在这边将两个同名参数的默认值设置为了 `3`

```python
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """

    # 设定输入层
    if channel_first:
        input_a, input_b =  Input(shape=(nc_in, None, None)), Input(shape=(nc_out, None, None))
    else:
        input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))
```

---

仍然记得使用 tf 时 `channel_axis` 设置成了 `-1`

> 两张图片的 Channel 维度被连接了起来，似乎意思指的就是，连接后的图片大小和原来一张的大小一样，但是有 RGB-RGB 这样的六个通道，视觉感受上看类似两张图片用一半透明度堆在了一起的感觉

```python
    _ = Concatenate(axis=channel_axis)([input_a, input_b])
```

---

使用上面预定义了初始化方法的 `conv2d` 函数来堆叠 `Conv2D` 层，参数含义如下：
- `ndf` 是 filters 的数量
- `kernel_size` 为 int 而不是 tuple 时，表示长宽一样
- `strides` 步长
- `padding` 填白的方法，这里使用了 `same` 让卷积之后图片的大小不变，参考 [CNN 基础概念理解](https://zhuanlan.zhihu.com/p/42559190)

激活函数使用 `LeakyReLU`, 参考[官方文档](https://keras.io/layers/advanced-activations/)

```python
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
```

---

这一部分是堆叠隐藏层。根据前面 `max_layers` 的说明是，它指定隐藏层的最大层数，因此这里就通过循环 `range(1, max_layers)` 来控制循环次数

两个遇到的新操作：
- `use_bias=False`, 这个参数指的是是否创建一个 bias vector 并且直接加到输出上，默认为 True, 因此似乎在第一层中使用，后面所有层都不使用
- `batchnorm()(_, training=1)` 是应用前面定义的 `BatchNormalization` 层，根据 Functional API 的说明，`batchnorm()` 的参数中，`_` 是前面那一层，但是

> `training=1` 参数是什么意思 没有找到

```python
    for layer in range(1, max_layers):
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name = 'pyramid.{0}'.format(layer)
                        ) (_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)
```

---

`out_feat` 参数就是 filters 的个数，通过第一层的 filters 和最大深度算出来

`ZeroPadding2D` 层的作用是把之前输出的图片上下左右各填充了一个空白像素，之后再送给下一个 `Conv2D` 层

在最终层中，激活函数用的是 `sigmoid`, 因为 Discriminator 最终做的是一个分类任务

> 这里 `name = 'final'.format(out_feat, 1)`, 不知道什么意思，字符串 'final' 中根本就没有 `{}` 符号来填充内容

```python
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1),
               activation = "sigmoid") (_)
    return Model(inputs=[input_a, input_b], outputs=_)
```

---

### Generator

> 推测的参数说明：

- `isize`: 图片的大小
- `nc_in` / `nc_out`: 图片的颜色通道，一般都还是用 `3`
- `ngf`: 第一层中 filters 的数量，对应 Discriminator 中的 `ndf`, `g` 和 `d` 分别指代 generator 和 discriminator
- `fixed_input_size`: 输入图像的大小是否是相同的

> `max_nf` 的含义是什么？推测是最大的 filters 数量，要看后面代码

```python
def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    max_nf = 8*ngf
```

---

`block` 是一个内部函数，它通过对自己递归来产生 Generator 中除了输入层和最后一个激活函数层之外的所有隐藏层

被调用：
- `block` 内部有一个递归调用
- `Input` 层后面紧跟了一个 `block` 函数，返回值后面连接了 `tanh` 激活函数层

推测的参数说明：

- `x`: `block` 产生的层连接着的上一个层
    - 首次调用传入了 `Input` 层
    - 递归调用传入了 `Conv2D` 后接的 `LeakyReLU` 层
- `s`: 一个控制堆叠层数的参数，让层数为 $\log_2 (\text{isize})$【或者`+1/-1`什么的，但是总体上说是这个数量级】
    - 首次调用传入了 `isize`
    - 递归调用传入了上一级递归的此参数的一半（取整）
- `nf_in`: `Conv2DTranspose` 所使用的 filters 的数量
    - 首次调用传入了 `nc_in`. 通过之前的基概知道，卷积层使用的 filters 的数量等价于其输出的图像的通道数，`Conv2DTranspose` 的作用是还原图像，所以输出的通道数要和输入图像的通道数相同
    - 递归调用传入了 `nf_next = min(nf_in*2, max_nf)`
- `use_batchnorm`: 是否使用神秘的 `BatchNormalization`, 说明前面有提到过
- `nf_out`: 输出时使用的 filters 的数量
    - 如果传入参数时不提供默认值，会被设置成和 `nf_in` 一样
    - 被调用的地方只有一个：`Conv2DTranspose` 的第一个参数
- `nf_next`: 传入 `block` 函数的层下一层连接的 `conv2d` 层所使用的 filters 的数量
    - 如果传入参数时不提供默认值，会被设置成 `nf_next = min(nf_in*2, max_nf)`, 其中 `nf_in` 是上一级的本参数，`max_nf` 是外层 `UNET_G` 第一行定义的 `max_nf = 8*ngf`


涉及到的 keras Layers 的说明：

- `Conv2DTranspose`: 即“反卷积”或“转置卷积”，是卷积的逆向过程（但是当然无法恢复所有信息），是 Generator 产生图像的关键

> 需要进一步搜集有关资料便于理解

- `Cropping2D`: 和之前的 `ZeroPadding2D` 作用相反，裁掉图片上下左右四个边缘

调用一次 `block` 函数，会堆叠一个 `Conv2D` 层和一个 `Conv2DTranspose` 层（以及其他的 cropping, padding, 激活函数层等等），以起到 Generator 的作用。并且通过递归，堆叠多层。

```python
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,
                            name = 'convt.{0}'.format(s))(x)
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
```

---

`block` 函数定义完成之后对其的调用，具体说明主要在上一段

```python
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])
```

---

以上部分完成了两个模型的定义

以下部分应该是编译和进行训练的步骤。

> $\lambda$ 和 `loadSize` 两个参数的含义不明
> 【为什么在 Python 代码里会用希腊字母形式的 $\lambda$ 作为变量名啊？？

```python
nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
λ = 10

loadSize = 286
imageSize = 256
batchSize = 1
lrD = 2e-4
lrG = 2e-4
```

---

构建 Discriminator 模型

```python
netD = BASIC_D(nc_in, nc_out, ndf)
netD.summary()
```

---

构建 Generator 模型

`SVG(model_to_dot....)` 这句运行不了，需要安装 `Graphviz`, 查了一下过程比较繁琐

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


netG = UNET_G(imageSize, nc_in, nc_out, ngf)
#SVG(model_to_dot(netG, show_shapes=True).create(prog='dot', format='svg'))
netG.summary()
```

---

```python
from keras.optimizers import RMSprop, SGD, Adam
```

---

运行 Notebook 后打印这里的 `real_A = netG.input` 和 `fake_B = netG.output`，分别得到了一个 `tf.Tensor`, 即 TensorFlow 张量。有关用法在官方文档中没有发现。

> tf 学的不是很深入，但是根据印象来说这个张量应该是动态的而不是静态的，可以理解为图里的一个节点。

`K` 是 `keras.backend`, `K.function()` 在官方文档中的说明是（直接复制黏贴了，链接页面过长也不好查找，并且没有 anchor ）：

> 具体作用和如何使用可以参考[这个 StackOverflow 回答](https://stackoverflow.com/questions/48142181/whats-the-purpose-of-keras-backend-function)以及[这些例子](https://www.programcreek.com/python/example/93732/keras.backend.function)，并且结合后面理解。

```text
keras.backend.function(inputs, outputs, updates=None)
Instantiates a Keras function.

Arguments:
- inputs: List of placeholder tensors.
- outputs: List of output tensors.
- updates: List of update ops.
- **kwargs: Passed to tf.Session.run.

Returns:
- Output values as Numpy arrays.

Raises:
- ValueError: if invalid kwargs are passed in.
```

```python
real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])
real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B])
output_D_fake = netD([real_A, fake_B])
```

---

```python
#loss_fn = lambda output, target : K.mean(K.binary_crossentropy(output, target))
loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))


loss_L1 = K.mean(K.abs(fake_B-real_B))
```


```python
loss_D = loss_D_real +loss_D_fake
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(netD.trainable_weights,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_D/2], training_updates)
```


```python
loss_G = loss_G_fake   + 100 * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(netG.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)


```


```python
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle

def load_data(file_pattern):
    return glob.glob(file_pattern)
def read_image(fn, direction=0):
    im = Image.open(fn)
    im = im.resize( (loadSize*2, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    imgA = arr[h1:h2, loadSize+w1:loadSize+w2, :]
    imgB = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        imgA=imgA[:,::-1]
        imgB=imgB[:,::-1]
    if channel_first:
        imgA = np.moveaxis(imgA, 2, 0)
        imgB = np.moveaxis(imgB, 2, 0)
    if direction==0:
        return imgA, imgB
    else:
        return imgB,imgA

data = "edges2shoes"
data = "facades"
direction = 0
trainAB = load_data('pix2pix/{}/train/*.jpg'.format(data))
valAB = load_data('pix2pix/{}/val/*.jpg'.format(data))
assert len(trainAB) and len(valAB)
```


```python
def minibatch(dataAB, batchsize, direction=0):
    length = len(dataAB)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(dataAB)
            i = 0
            epoch+=1
        dataA = []
        dataB = []
        for j in range(i,i+size):
            imgA,imgB = read_image(dataAB[j], direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i+=size
        tmpsize = yield epoch, dataA, dataB

```


```python
from IPython.display import display
def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    display(Image.fromarray(int_X))
```


```python
train_batch = minibatch(trainAB, 6, direction=direction)
_, trainA, trainB = next(train_batch)
showX(trainA)
showX(trainB)
del train_batch, trainA, trainB
```


```python
def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i+1]])[0] for i in range(A.shape[0])], axis=0)
```


```python
import time
from IPython.display import clear_output
t0 = time.time()
niter = 50
gen_iterations = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0

display_iters = 500
val_batch = minibatch(valAB, 6, direction)
train_batch = minibatch(trainAB, batchSize, direction)

while epoch < niter:
    epoch, trainA, trainB = next(train_batch)
    errD,  = netD_train([trainA, trainB])
    errD_sum +=errD

    errG, errL1 = netG_train([trainA, trainB])
    errG_sum += errG
    errL1_sum += errL1
    gen_iterations+=1
    if gen_iterations%display_iters==0:
        if gen_iterations%(5*display_iters)==0:
            clear_output()
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_L1: %f'
        % (epoch, niter, gen_iterations, errD_sum/display_iters, errG_sum/display_iters, errL1_sum/display_iters), time.time()-t0)
        _, valA, valB = train_batch.send(6)
        fakeB = netG_gen(valA)
        showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
        errL1_sum = errG_sum = errD_sum = 0
        _, valA, valB = next(val_batch)
        fakeB = netG_gen(valA)
        showX(np.concatenate([valA, valB, fakeB], axis=0), 3)

```


```python
_, valA, valB = train_batch.send(6)
fakeB = netG_gen(valA)
showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
errL1_sum = errG_sum = errD_sum = 0
_, valA, valB = next(val_batch)
fakeB = netG_gen(valA)
showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
```
