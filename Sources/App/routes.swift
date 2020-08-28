import Vapor
import TensorFlow

// import Python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

struct Data: Content {
    let row: [Float] // input data (array of floats)
}

struct Response: Content {
    let request: Data        // original input data (array)
    let predictions: [Float] // output predictions
    let classes: [String]    // ML classes
}

func routes(_ app: Application) throws {
    app.get { req in
        return "It works!"
    }

    app.get("hello") { req -> String in
        return "Hello, world!"
    }

    app.post("inference") { req -> Response in
        let data = try req.content.decode(Data.self)
        let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]
        let np = Python.import("numpy")
        let numpyArray = np.array([data.row], dtype: np.float32)
        let tfData: Tensor<Float> = Tensor<Float>(numpy: numpyArray)!
        let preds = model(tfData)
        printPredictions(classNames: classNames, preds: model(tfData))
        let out = outputPredictions(classNames: classNames, preds: preds)
        return Response(request: data, predictions: out, classes: classNames)
    }
}
