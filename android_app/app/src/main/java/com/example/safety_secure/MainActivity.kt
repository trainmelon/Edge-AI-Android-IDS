package com.example.safety_secure
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.graphics.Color
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView
import androidx.activity.ComponentActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.net.Inet4Address
import java.net.NetworkInterface
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.timerTask

class MainActivity : ComponentActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvLog: TextView
    private lateinit var tvMode: TextView
    private lateinit var tvRealNetwork: TextView
    private lateinit var progressThreat: ProgressBar
    private lateinit var scrollView: ScrollView
    private lateinit var switchMode: Switch
    private lateinit var btnHistory: Button

    private lateinit var sharedPrefs: SharedPreferences

    private var tflite: Interpreter? = null
    private var logBuilder = StringBuilder()
    private var timer: Timer? = null

    private var detectionInterval: Long = 1500L

    // 🏆 新增：5大分类映射字典
    private val attackNames = arrayOf(
        "✅ 正常流量",
        "🚨 DoS (拒绝服务)",
        "🚨 Probe (探测嗅探)",
        "🚨 R2L (越权访问)",
        "🚨 U2R (非法提权)"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tv_status)
        tvLog = findViewById(R.id.tv_log)
        tvMode = findViewById(R.id.tv_mode)
        tvRealNetwork = findViewById(R.id.tv_real_network)
        progressThreat = findViewById(R.id.progress_threat)
        scrollView = findViewById(R.id.scroll_view)
        switchMode = findViewById(R.id.switch_mode)
        btnHistory = findViewById(R.id.btn_history)

        sharedPrefs = getSharedPreferences("IDS_History", Context.MODE_PRIVATE)

        detectRealNetworkEnvironment()

        try {
            // 🎯 修改点 1：加载全新的多分类模型
            tflite = Interpreter(loadModelFile())
            appendLog("✅ 5分类多模态 TFLite 引擎加载成功！\n等待网络流量接入...")
        } catch (e: Exception) {
            appendLog("❌ 模型加载失败: ${e.message}")
            return
        }

        switchMode.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                detectionInterval = 1500L
                tvMode.text = "智能调度：当前处于 [高性能模式]"
                tvMode.setTextColor(Color.parseColor("#CCCCCC"))
            } else {
                detectionInterval = 5000L
                tvMode.text = "智能调度：当前处于 [低功耗休眠模式]"
                tvMode.setTextColor(Color.parseColor("#FFC107"))
            }
            restartPipeline()
        }

        btnHistory.setOnClickListener {
            showHistoryDialog()
        }

        startDetectionPipeline()
    }

    private fun detectRealNetworkEnvironment() {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork
        val capabilities = cm.getNetworkCapabilities(network)

        val netType = when {
            capabilities == null -> "未连接"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "WLAN (Wi-Fi)"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "移动蜂窝数据"
            else -> "未知网络"
        }

        var localIp = "获取失败"
        try {
            val en = NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val intf = en.nextElement()
                val enumIpAddr = intf.inetAddresses
                while (enumIpAddr.hasMoreElements()) {
                    val inetAddress = enumIpAddr.nextElement()
                    if (!inetAddress.isLoopbackAddress && inetAddress is Inet4Address) {
                        localIp = inetAddress.hostAddress?.toString() ?: "未知"
                    }
                }
            }
        } catch (e: Exception) { e.printStackTrace() }

        tvRealNetwork.text = "📡 物理探针 | 环境: $netType | 本机IP: $localIp"
    }

    private fun loadModelFile(): MappedByteBuffer {
        // 🎯 确保文件名与刚才放入 assets 的文件名完全一致！
        val fileDescriptor = assets.openFd("Multiclass_IDS_Quantized.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun restartPipeline() {
        timer?.cancel()
        startDetectionPipeline()
    }

    private fun startDetectionPipeline() {
        timer = Timer()
        timer?.schedule(timerTask {
            // 模拟生成特征：为了让模型在演示时能输出不同的攻击类型，我们对异常数据增加随机剧烈波动
            val isAttack = Random().nextInt(100) > 85
            val attackTypeSim = Random().nextInt(4) + 1 // 模拟1-4的波动源

            val inputFeatures = Array(1) { FloatArray(41) }
            for (i in 0 until 41) {
                inputFeatures[0][i] = if (isAttack) (Random().nextFloat() * 2f + attackTypeSim) else (Random().nextFloat() * 0.5f)
            }

            // 🎯 修改点 2：接收数组从 2 个变成了 5 个！
            val outputProbabilities = Array(1) { FloatArray(5) }
            tflite?.run(inputFeatures, outputProbabilities)

            // 🎯 修改点 3：寻找概率最高的那个分类 (ArgMax 逻辑)
            var maxProb = 0f
            var predictedClassIndex = 0
            for (i in 0 until 5) {
                if (outputProbabilities[0][i] > maxProb) {
                    maxProb = outputProbabilities[0][i]
                    predictedClassIndex = i
                }
            }

            // 转换为百分比
            val confidenceScore = maxProb * 100

            runOnUiThread {
                updateUI(predictedClassIndex, confidenceScore)
            }
        }, 1000, detectionInterval)
    }

    // 🎯 修改点 4：根据具体的攻击类别索引更新 UI
    private fun updateUI(predictedClassIndex: Int, confidenceScore: Float) {
        val time = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val scoreInt = confidenceScore.toInt()

        // 我们用最高概率的置信度作为进度条显示
        progressThreat.progress = scoreInt

        val specificAttackName = attackNames[predictedClassIndex]

        if (predictedClassIndex > 0) {
            // 索引大于 0，说明是攻击 (DoS, Probe, R2L, U2R 之一)
            tvStatus.text = "⚠️ 拦截: ${specificAttackName.replace("🚨 ", "")}"
            tvStatus.setTextColor(Color.parseColor("#FF4B4B"))
            progressThreat.progressTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#FF4B4B"))

            val logMsg = "[$time] 阻断攻击! 类型:[$specificAttackName] 置信度:$scoreInt%"
            appendLog(logMsg)
            saveToHistory(logMsg)
        } else {
            // 索引为 0，正常流量
            tvStatus.text = "✅ 网络环境安全"
            tvStatus.setTextColor(Color.parseColor("#00FF00"))
            progressThreat.progressTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#00FF00"))
            appendLog("[$time] 流量包检测通过 (正常置信度:$scoreInt%)")
        }
    }

    private fun saveToHistory(record: String) {
        val currentHistory = sharedPrefs.getString("history_logs", "")
        val newHistory = (record + "\n\n" + currentHistory).take(1500)
        sharedPrefs.edit().putString("history_logs", newHistory).apply()
    }

    private fun showHistoryDialog() {
        val historyData = sharedPrefs.getString("history_logs", "暂无历史告警记录。")
        AlertDialog.Builder(this)
            .setTitle("🚨 历史高危告警记录")
            .setMessage(if (historyData.isNullOrBlank()) "暂无历史告警记录。" else historyData)
            .setPositiveButton("关闭", null)
            .setNegativeButton("清空记录") { _, _ ->
                sharedPrefs.edit().clear().apply()
            }
            .setNeutralButton("一键导出") { _, _ ->
                if (!historyData.isNullOrBlank()) {
                    exportLogsToWeChat(historyData)
                }
            }
            .show()
    }

    private fun exportLogsToWeChat(logs: String) {
        val sendIntent: Intent = Intent().apply {
            action = Intent.ACTION_SEND
            putExtra(Intent.EXTRA_TEXT, "【移动端入侵防御系统 - 拦截日志导出】\n\n$logs")
            type = "text/plain"
        }
        val shareIntent = Intent.createChooser(sendIntent, "导出防御日志至...")
        startActivity(shareIntent)
    }

    private fun appendLog(msg: String) {
        logBuilder.insert(0, msg + "\n")
        if (logBuilder.length > 3000) {
            logBuilder.setLength(3000)
        }
        tvLog.text = logBuilder.toString()
    }

    override fun onDestroy() {
        super.onDestroy()
        timer?.cancel()
        tflite?.close()
    }
}