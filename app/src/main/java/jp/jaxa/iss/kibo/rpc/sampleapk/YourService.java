package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
//    Interpreter interpreter;
//    final int inputSize = 640;
//    final float threshold = 0.4f;
//    final String[] CLASS_NAMES = {
//            "coin", "compass", "coral", "crystal", "diamond", "emerald", "fossil", "key", "letter", "shell", "treasure_box"
//    };

    @Override
    protected void runPlan1(){
        // nggo mlaku
        api.startMission();

//        AssetManager assetManager = getAssets();
//        try {
//            MappedByteBuffer modelBuffer = loadModelFile(assetManager, "model2.tflite");
//            Log.i("AR Detection", "Alamak berhasil");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        Point point = new Point(11.42d, -9.92284d, 5.5d);
        Quaternion quaternion = eulerToQuaternion_use(0, 0, -90);
        api.moveTo(point, quaternion, true);
        SystemClock.sleep(3000);
        ARResult result = AR_cropping(1, 0);
        ObjectDetector.ObjectDetectorOptions options =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setMaxResults(3)
                        .setScoreThreshold(0.5f)
                        .build();
        try {
            ObjectDetector objectDetector = ObjectDetector.createFromFileAndOptions(
                    this,
                    "model2.tflite",options);
            Bitmap bitmap = BitmapFactory.decodeFile("/storage/emulated/0/data/jp.jaxa.iss.kibo.rpc.sampleapk/immediate/DebugImages/post_1.png");
            TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
            List<Detection> results = objectDetector.detect(tensorImage);
            for (Detection detection : results) {
                RectF box = detection.getBoundingBox();
                Category category = detection.getCategories().get(0);
                Log.d("DETECT", category.getLabel() + ": " + category.getScore());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.i("AR Detection", "AR Complete: " + result.arComplete + ", Marker Count: " + result.detectedIdsCount);

        point = new Point(11.292d, -8.85d, 4.5d);
        quaternion = eulerToQuaternion_use(0, 90, 0);
        api.moveTo(point, quaternion, true);
        SystemClock.sleep(3000);
        ARResult result2 = AR_cropping(2, 0);
        Log.i("AR Detection", "AR Complete: " + result2.arComplete + ", Marker Count: " + result2.detectedIdsCount);

        Mat image2 = api.getMatNavCam();
        api.saveMatImage(image2,"area_2.png");
        api.setAreaInfo(2, "item_name", 2);

        point = new Point(11.05d, -7.8d, 4.5d);
        quaternion = eulerToQuaternion_use(0, 90, 0);
        api.moveTo(point, quaternion, true);
        SystemClock.sleep(3000);
        ARResult result3 = AR_cropping(3, 0);
        Log.i("AR Detection", "AR Complete: " + result3.arComplete + ", Marker Count: " + result3.detectedIdsCount);

        Mat image3 = api.getMatNavCam();
        api.saveMatImage(image3,"area_3.png");
        api.setAreaInfo(3, "item_name", 3);

        point = new Point(10.7d, -6.71d, 4.8d);
        quaternion = eulerToQuaternion_use(0, 0, -180);
        api.moveTo(point, quaternion, true);
        SystemClock.sleep(3000);
        ARResult result4 = AR_cropping(4, 0);
        Log.i("AR Detection", "AR Complete: " + result4.arComplete + ", Marker Count: " + result4.detectedIdsCount);

        Mat image4 = api.getMatNavCam();
        api.saveMatImage(image4,"area_4.png");
        api.setAreaInfo(4, "item_name", 4);

        point = new Point(11.143d, -6.7607d, 4.7654d);
        quaternion = eulerToQuaternion_use(0, 0, 90);
        api.moveTo(point, quaternion, true);
        SystemClock.sleep(3000);
        ARResult resultAstronaut = AR_cropping(5, 0);
        Log.i("AR Detection", "AR Complete: " + resultAstronaut.arComplete + ", Marker Count: " + resultAstronaut.detectedIdsCount);

        api.reportRoundingCompletion();
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public ARResult AR_cropping (int target_num,int emr){ //meiji//
        int ar_complete =0;
        boolean check=false;
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        kernel.put(0, 0, 0);
        kernel.put(0, 1, -1);
        kernel.put(0, 2, 0);
        kernel.put(1, 0, -1);
        kernel.put(1, 1, 5);
        kernel.put(1, 2, -1);
        kernel.put(2, 0, 0);
        kernel.put(2, 1, -1);
        kernel.put(2, 2, 0);
        double[][] cameraParam = api.getNavCamIntrinsics();
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_32FC1);
        Mat dstMatrix = new Mat(1, 5, CvType.CV_32FC1);
        cameraMatrix.put(0, 0, cameraParam[0]);
        dstMatrix.put(0, 0, cameraParam[1]);
        Mat src = new Mat();
        Mat sharpened= new Mat();
        Mat undistort=new Mat();
        if(target_num==5){
            src =api.getMatDockCam() ;
        }else {
            src = api.getMatNavCam();
        }

        Calib3d.undistort(src, undistort, cameraMatrix, dstMatrix);
        if(emr ==1){
            undistort=src;

        }
        Imgproc.filter2D(undistort, sharpened, -1, kernel);
        api.saveMatImage(sharpened, "Pre-" + target_num + ".png");
        Dictionary arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        Mat ids = new Mat();

        List<Mat> corners = new ArrayList<>();


        Aruco.detectMarkers(sharpened, arucoDict, corners, ids);
        Log.i("ids count","ids counting :"+ ids.rows());


        if (!corners.isEmpty() && check == false) {
            float markerLength = 0.05f;
            // Find the marker with the desired y-coordinate
            int selectedIndex = 0;
            double selectedY = corners.get(0).get(0, 1)[1];  // y-coordinate of the first corner of the first marker

            for (int i = 1; i < corners.size(); i++) {
                double y = corners.get(i).get(0, 1)[1];
                if ((target_num == 2 && y > selectedY) || (target_num == 3 && y < selectedY)) {
                    selectedY = y;
                    selectedIndex = i;
                }
            }

            // Use the marker with the desired y-coordinate for further processing
            Mat selectedCorner = corners.get(selectedIndex);
            int selectedId = (int) ids.get(selectedIndex, 0)[0];
            List<Mat> selectedCorners = new ArrayList<>();
            selectedCorners.add(selectedCorner);
            // Estimate pose of the selected marker
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            Aruco.estimatePoseSingleMarkers(selectedCorners, markerLength, cameraMatrix, dstMatrix, rvecs, tvecs);

            MatOfPoint2f cornerPoints = new MatOfPoint2f(selectedCorner);
            org.opencv.core.Point[] cornerArray = cornerPoints.toArray();

            // Calculate the Euclidean distances
            double pixelDistance1 = Core.norm(new MatOfPoint2f(cornerArray[0]), new MatOfPoint2f(cornerArray[1]));
            double pixelDistance2 = Core.norm(new MatOfPoint2f(cornerArray[0]), new MatOfPoint2f(cornerArray[3]));
            double pixelDistance3 = Core.norm(new MatOfPoint2f(cornerArray[1]), new MatOfPoint2f(cornerArray[2]));
            double pixelDistance4 = Core.norm(new MatOfPoint2f(cornerArray[2]), new MatOfPoint2f(cornerArray[3]));
            double pixelDistance = (pixelDistance1 + pixelDistance2 + pixelDistance3 + pixelDistance4) / 4;

            // Calculate the ratio
            double pixelToMRatio = pixelDistance / markerLength;


            // Print the pose information
            double TL = pixelToMRatio * 0.2375;
            double TR = pixelToMRatio * -0.0375;
            double TH = pixelToMRatio * 0.0375;
            double BH = pixelToMRatio * 0.1125;
            double angle = Math.atan2(cornerArray[0].y - cornerArray[1].y, cornerArray[0].x - cornerArray[1].x);
            double angleDegrees = Math.toDegrees(angle);
            org.opencv.core.Point center = new org.opencv.core.Point(
                    (cornerArray[0].x + cornerArray[2].x) / 2.0,
                    (cornerArray[0].y + cornerArray[2].y) / 2.0
            );

            Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angleDegrees, -1.0);
            Mat rotatedImage = new Mat();
            Imgproc.warpAffine(sharpened, rotatedImage, rotationMatrix, undistort.size());

            // Compute the new corner points after rotation
            MatOfPoint2f newCorners = new MatOfPoint2f();
            Core.perspectiveTransform(cornerPoints, newCorners, rotationMatrix);
            org.opencv.core.Point[] newCornerArray = newCorners.toArray();
            int centerX = (int) center.x;
            int centerY = (int) center.y;
            Log.i("AR_pic(coordinate):", Integer.toString(centerX) + ',' + Integer.toString(centerY));
            int xMin = (int) (centerX - TL);
            int yMin = (int) (centerY - TH);
            int xMax = (int) (centerX + TR);
            int yMax = (int) (centerY + BH);

            // Ensure the coordinates are within image bounds
            xMin = Math.max(0, xMin);
            yMin = Math.max(0, yMin);
            xMax = Math.min(rotatedImage.width(), xMax);
            yMax = Math.min(rotatedImage.height(), yMax);

            // Crop the image
            Rect roi = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
            Mat croppedImage = new Mat(rotatedImage, roi);
            ar_complete=target_num;
            check=true;
            api.saveMatImage(croppedImage,"post_" + target_num + ".png");

        }else if(!check && emr != 1){
            Log.i("AR", "not found, retrying once");
            return AR_cropping(target_num, 1); // retry sekali saja
        }else if (!check && emr == 1){
            Log.i("AR", "still not found after retry.");
        }

        return new ARResult(ar_complete, ids.rows());
    }

    public  Quaternion eulerToQuaternion_use(double x, double y, double z) { //meiji//
        double yaw = Math.toRadians(z); //radian = degree*PI/180
        double pitch = Math.toRadians(y);
        double roll = Math.toRadians(x);

        double cy = Math.cos(yaw * 0.5);
        double sy = Math.sin(yaw * 0.5);
        double cp = Math.cos(pitch * 0.5);
        double sp = Math.sin(pitch * 0.5);
        double cr = Math.cos(roll * 0.5);
        double sr = Math.sin(roll * 0.5);

        double qx = sr * cp * cy - cr * sp * sy;
        double qy = cr * sp * cy + sr * cp * sy;
        double qz = cr * cp * sy - sr * sp * cy;
        double qw = cr * cp * cy + sr * sp * sy;

        return new Quaternion((float) qx, (float) qy, (float) qz, (float) qw);
    }



    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }

    // You can add your method.
    private String yourMethod(){
        return "your method";
    }

//    public List<Detection> detect(String filePath) {
//        File imgFile = new File(filePath);
//        Bitmap bitmap;
//        if (imgFile.exists()) {
//            bitmap =  BitmapFactory.decodeFile(imgFile.getAbsolutePath());
//        }
//        else{return null;}
//
//        Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
//        TensorImage inputImage = TensorImage.fromBitmap(resized);
//
//        float[][][] output = new float[1][25200][16]; // adjust based on model output
//        interpreter.run(inputImage.getBuffer(), output);
//
//        List<Detection> detections = processOutput(output[0], bitmap.getWidth(), bitmap.getHeight());
//
//        return detections;
//    }
//
//    private List<Detection> processOutput(float[][] output, int origWidth, int origHeight) {
//        List<Detection> detections = new ArrayList<>();
//
//        for (float[] pred : output) {
//            float conf = pred[4];
//            if (conf < threshold) continue;
//
//            float[] classProbs = new float[output[0].length - 5];
//            System.arraycopy(pred, 5, classProbs, 0, classProbs.length);
//            int classId = argMax(classProbs);
//            float classScore = classProbs[classId] * conf;
//            if (classScore < threshold) continue;
//
//            // Convert from xywh to xyxy
//            float cx = pred[0] * origWidth;
//            float cy = pred[1] * origHeight;
//            float w = pred[2] * origWidth;
//            float h = pred[3] * origHeight;
//            float xmin = cx - w / 2;
//            float ymin = cy - h / 2;
//            float xmax = cx + w / 2;
//            float ymax = cy + h / 2;
//
//            detections.add(new Detection(classId, classScore, new RectF(xmin, ymin, xmax, ymax)));
//        }
//
//        return detections;
//    }

    private int argMax(float[] array) {
        int maxIdx = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

//    private static class Detection {
//        int classId;
//        float score;
//        RectF box;
//
//        Detection(int classId, float score, RectF box) {
//            this.classId = classId;
//            this.score = score;
//            this.box = box;
//        }
//    }
}
