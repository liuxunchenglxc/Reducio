package com.main.tflmodelruntimeevaluation;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.text.Layout;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.util.Objects;

import okhttp3.Headers;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;


public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("runtimeevaluation");
    }

    //读写权限
    private static final String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE"};

    public static void verifyStoragePermissions(Activity activity) {
        try {
            //检测是否有写的权限
            for (String s : PERMISSIONS_STORAGE) {
                int permission = ActivityCompat.checkSelfPermission(activity, s);
                if (permission != PackageManager.PERMISSION_GRANTED) {
                    // 没有写的权限，去申请写的权限，会弹出对话框
                    ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //请求状态码
    private static final int REQUEST_PERMISSION_CODE = 1;

    private final OkHttpClient client = new OkHttpClient();

    public byte[] getALite(String hostPort, String job) throws Exception {
        Request request = new Request.Builder()
                .url("http://" + hostPort + "/get_a_lite/" + job)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            Headers responseHeaders = response.headers();
            for (int i = 0; i < responseHeaders.size(); i++) {
                System.out.println(responseHeaders.name(i) + ": " + responseHeaders.value(i));
            }
            return Objects.requireNonNull(response.body()).bytes();
        }
    }

    public String getAJob(String hostPort, String chip) throws Exception {
        Request request = new Request.Builder()
                .url("http://" + hostPort + "/get_a_job/" + chip)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            Headers responseHeaders = response.headers();
            for (int i = 0; i < responseHeaders.size(); i++) {
                System.out.println(responseHeaders.name(i) + ": " + responseHeaders.value(i));
            }
            return Objects.requireNonNull(response.body()).string();
        }
    }

    public String doneAJob(String hostPort, String chip, String job, String status, String time) throws Exception {
        Request request = new Request.Builder()
                .url("http://" + hostPort + "/done_a_job/" + chip + "/" + job + "/" + status + "/" + time)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            Headers responseHeaders = response.headers();
            for (int i = 0; i < responseHeaders.size(); i++) {
                System.out.println(responseHeaders.name(i) + ": " + responseHeaders.value(i));
            }
            return Objects.requireNonNull(response.body()).string();
        }
    }

    final LogHandler logHandler = new LogHandler(this);

    public void logMsg(String log) {
        Bundle bd = new Bundle();
        bd.putString("log", log);
        Message msg = new Message();
        msg.setData(bd);
        logHandler.sendMessage(msg);
    }

    Button buttonOpen;
    Button buttonRun;
    Button buttonDown;
    TextView textPath;
    TextView textTime;
    TextView textLog;
    ScrollView scrollView;
    EditText editHost;
    EditText editChip;
    Button buttonGo;
    Button buttonStop;
    Button buttonHW;
    Button buttonGT;
    Button buttonLFK;

    int isGo = 1;

    private int getTextViewHeight(TextView view) {
        Layout layout = view.getLayout();
        int desired = layout.getLineTop(view.getLineCount());
        int padding = view.getCompoundPaddingTop() + view.getCompoundPaddingBottom();
        return desired + padding;
    }

    @SuppressLint("HandlerLeak")
    private class LogHandler extends Handler {

        //弱引用持有HandlerActivity , GC 回收时会被回收掉
        private final WeakReference weakReference;

        public LogHandler(MainActivity activity) {
            this.weakReference = new WeakReference(activity);
        }

        @SuppressLint("SetTextI18n")
        @Override
        public void handleMessage(Message msg) {
            Object activity = weakReference.get();
            super.handleMessage(msg);
            if (null != activity) {
                String log = msg.getData().getString("log");
                textLog.setText(textLog.getText() + "\n" + log);
                textLog.post(new Runnable() {
                    @Override
                    public void run() {
                        int scrollAmount = textLog.getLayout().getLineTop(textLog.getLineCount()) - textLog.getHeight();
                        textLog.scrollTo(0, Math.max(scrollAmount, 0));
                    }
                });
            }
        }
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        verifyStoragePermissions(this);
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }
        textPath = findViewById(R.id.text);
        textTime = findViewById(R.id.text2);
        textLog = findViewById(R.id.textView3);
        scrollView = findViewById(R.id.scroll);
        textLog.setMovementMethod(ScrollingMovementMethod.getInstance());
        editChip = findViewById(R.id.editText2);
        editHost = findViewById(R.id.editText);
        buttonGo = findViewById(R.id.button4);
        buttonStop = findViewById(R.id.button6);
        buttonHW = findViewById(R.id.button7);
        buttonGT = findViewById(R.id.button8);
        buttonLFK = findViewById(R.id.button9);
        buttonGo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                final String HostPort = editHost.getText().toString();
                final String ChipName = editChip.getText().toString();
                isGo = 1;
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        String cachePath = getCacheDir().getPath();
                        int isJob = 0;
                        String job_t = "";
                        while (isGo == 1) {
                            try {
                                isJob = 0;
                                logMsg("Client呼叫Host请回答,是否有任务?");
                                String jobCode = getAJob(HostPort, ChipName);
                                String jobStatus = jobCode.split(":")[0];
                                if (Objects.equals(jobStatus, "HAS")) {
                                    String job = jobCode.split(":")[1];
                                    logMsg("Host收到,任务:" + job + "，请执行.");
                                    logMsg("Client收到,执行任务.");
                                    isJob = 1;
                                    job_t = job;
                                    byte[] lite = getALite(HostPort, job);
                                    logMsg("Client数据接收正常.");
                                    FileOutputStream out = new FileOutputStream(cachePath + "/temp.tflite");
                                    out.write(lite);
                                    out.close();
                                    logMsg("Client文件写入正常.");
                                    isJob = 2;
                                    double time = RuntimeEvaluation.runtimeEvaluate(cachePath + "/temp.tflite", cachePath);
                                    logMsg("Client模型测试正常.");
                                    String stime = String.valueOf(time);
                                    logMsg("Client报告任务:" + job + ",平均推理时间:" + stime);
                                    SystemClock.sleep(1000);
                                    while (true) {
                                        String reportCode = doneAJob(HostPort, ChipName, job, "OK", stime);
                                        if (Objects.equals(reportCode, "OK")) {
                                            logMsg("Host收到.");
                                            break;
                                        } else {
                                            int dtime = 1000;
                                            logMsg("Client等待" + String.valueOf(dtime) + "毫秒重新报告任务.");
                                            SystemClock.sleep(dtime);
                                        }
                                    }
                                } else {
                                    logMsg("Host收到,暂无任务,继续等待.");
                                    int time = 60000;
                                    logMsg("Client收到,等待" + String.valueOf(time) + "毫秒.");
                                    SystemClock.sleep(time);
                                }
                            } catch (Exception e) {
                                logMsg("Client异常.");
                                try {
                                    if (isJob == 1) {
                                        logMsg("Client报告任务:" + job_t + ",其他异常,存在恢复可能");
                                        while (true) {
                                            String reportCode = null;
                                            reportCode = doneAJob(HostPort, ChipName, job_t, "ERRO", "0");
                                            if (Objects.equals(reportCode, "OK")) {
                                                logMsg("Host收到.");
                                                break;
                                            } else {
                                                int dtime = 1000;
                                                logMsg("Client等待" + String.valueOf(dtime) + "毫秒重新报告任务.");
                                                SystemClock.sleep(dtime);
                                            }
                                        }
                                    }
                                    else if (isJob == 2) {
                                        logMsg("Client报告任务:" + job_t + ",推理异常");
                                        while (true) {
                                            String reportCode = null;
                                            reportCode = doneAJob(HostPort, ChipName, job_t, "ERRI", "0");
                                            if (Objects.equals(reportCode, "OK")) {
                                                logMsg("Host收到.");
                                                break;
                                            } else {
                                                int dtime = 1000;
                                                logMsg("Client等待" + String.valueOf(dtime) + "毫秒重新报告任务.");
                                                SystemClock.sleep(dtime);
                                            }
                                        }
                                    }
                                    else {
                                        e.printStackTrace();
                                        int dtime = 1000;
                                        SystemClock.sleep(dtime);
                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                }
                                e.printStackTrace();
                                int dtime = 1000;
                                SystemClock.sleep(dtime);
                            }
                        }
                    }
                }).start();
            }
        });
        buttonStop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isGo = 0;
            }
        });
        buttonHW.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                editChip.setText("Huawei Kyrin 9000");
            }
        });
        buttonGT.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                editChip.setText("Qualcomm Snapdragon 865");
            }
        });
        buttonLFK.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                editChip.setText("MediaTek Dimensity 9000+");
            }
        });
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == Activity.RESULT_OK) {
            Uri uri = data.getData();
            try {
                @SuppressLint("Recycle") Cursor returnCursor = this.getContentResolver().query(uri, null, null, null, null);
                int nameIndex = returnCursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                returnCursor.moveToFirst();
                String name = (returnCursor.getString(nameIndex));
                File file = new File(this.getFilesDir(), name);
                InputStream inputStream = null;
                try {
                    inputStream = this.getContentResolver().openInputStream(uri);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                FileOutputStream outputStream = new FileOutputStream(file);
                int read = 0;
                int maxBufferSize = 1 * 1024 * 1024;
                int bytesAvailable = inputStream.available();
                int bufferSize = Math.min(bytesAvailable, maxBufferSize);
                final byte[] buffers = new byte[bufferSize];
                while ((read = inputStream.read(buffers)) != -1) {
                    outputStream.write(buffers, 0, read);
                }
                returnCursor.close();
                inputStream.close();
                outputStream.close();
                if (file.canRead()) {
                    Log.i("MainActivity", "Readable");
                } else {
                    Log.i("MainActivity", "Unreadable");
                }
                textPath.setText(file.getPath());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (int i = 0; i < permissions.length; i++) {
                Log.i("MainActivity", "申请的权限为：" + permissions[i] + ",申请结果：" + grantResults[i]);
            }
        }
    }
}