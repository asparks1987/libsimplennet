package ai.simplennet;

import java.nio.file.Path;

public final class SmokeTest {
    public static void main(String[] args) {
        SimplePredictor predictor = new SimplePredictor(OutputType.INT);
        predictor.fit(new double[][]{{0}, {1}, {2}, {3}}, new double[]{0, 2, 4, 6});
        int[] values = predictor.predictInts(new double[][]{{4}});
        if (values.length != 1) {
            throw new AssertionError("expected one prediction");
        }
        Path model = Path.of("java-smoke.snet");
        predictor.save(model);
        SimplePredictor loaded = SimplePredictor.load(model);
        int[] reloaded = loaded.predictInts(new double[][]{{4}});
        if (reloaded.length != 1) {
            throw new AssertionError("expected one prediction after load");
        }
        predictor.close();
        loaded.close();
        System.out.println("java smoke test passed");
    }
}
