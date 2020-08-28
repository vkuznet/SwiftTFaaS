### SwiftTFaaS
This repository contains work in progress towards Swift TensorFlow as a
Service. The idea is similar to [TFaaS](https://github.com/vkuznet/TFaaS)
with implementation of web server and TF components in Swift.
The web framework is based on [Vapor](https://github.com/vapor/vapor).

For TF model we use model produced by
[SwiftTFExample](https://github.com/vkuznet/SwiftMLExample).

### Build notes
In order to start with Vapor you need to build its toolbox.
```
# NOTE: to build vapor please use Apple/Linux vanila swift toolchain
#       do not use TF toolchain

# download vapor toolbox
git clone https://github.com/vapor/toolbox.git
cd toolbox
# optional you may check out particular branch/tag
git checkout 18.2.2
# build vapor tool
swift build -c release --disable-sandbox --enable-test-discovery --verbose
# copy vapor executable to your favorite OS location
sudo cp .build/release/vapor /usr/local/bin
```
Now, we can setup a new project (this one is already done):
```
vapor new <YourProjectName>
```
and, start coding. To build the code just use
```
swift build
```
from your project area.

### How to run and access the web server
To run the new code please use
```
swift run
```
The server will start on port 8080.

You may add new area `Model` to your project where you can store
your TF models, e.g. when I added `model.tf.npy` TF model file it
appears at a load time
```
# next the following will appear on your screen
loading weights from: model/model.tf.npy
[ NOTICE ] Server starting on http://127.0.0.1:8080
```
To place a new request please use
```
curl http://localhost:8080/inference -H "Content-Type: application/json" -d '{"row":[5.9, 3.0, 4.2, 1.5]}'
```
The response will look like:
```
{"request":{"row":[5.0999999046325684,3.2999999523162842,1.7000000476837158,0.5]},"classes":["Iris setosa","Iris versicolor","Iris virginica"],"predictions":[0.84249889850616455,0.091388963162899017,0.066112160682678223]}
```
