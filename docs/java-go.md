# Java And Go

Java and Go are runtime-supported through the shared native DLL.

## Java

The Java wrapper lives in `java/src/main/java/ai/simplennet`.

```java
SimplePredictor predictor = new SimplePredictor(OutputType.INT);
predictor.fit(inputs, targets);
int[] values = predictor.predictInts(inputs);
```

Run with:

```bash
java -Djava.library.path=build-native/Release -cp java/target/classes ai.simplennet.SmokeTest
```

## Go

The Go wrapper lives in `go/simplennet`.

```go
predictor, err := simplennet.NewSimplePredictor(simplennet.OutputInt)
if err != nil {
    panic(err)
}
defer predictor.Close()
_ = predictor.Fit(inputs, targets)
values, _ := predictor.PredictInts(inputs)
```

Set the native DLL path before running Go tests:

```powershell
$env:SIMPLENET_NATIVE_LIBRARY = "D:\path\to\build-native\Release\simplennet_native.dll"
go test ./...
```

## Binding Boundary

Both wrappers use the shared native DLL so all languages share one native training and prediction implementation.
