package jp.jaxa.iss.kibo.rpc.sampleapk;

public class ARResult {
    public int arComplete;
    public int detectedIdsCount;

    public ARResult(int arComplete, int detectedIdsCount) {
        this.arComplete = arComplete;
        this.detectedIdsCount = detectedIdsCount;
    }
}