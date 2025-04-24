const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const path = require("path");
const sendVerificationEmail = require("./emailService");
const verificationCodes = new Map();

const app = express();
const PORT = process.env.PORT || 10000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// MongoDB connection
mongoose.connect("mongodb+srv://abiram:abi18@cluster0.tklkrqn.mongodb.net/deviceFP?retryWrites=true&w=majority&appName=Cluster0")
  .then(() => console.log("✅ MongoDB connected"))
  .catch((err) => console.error("❌ MongoDB connection error:", err));

// Schemas
const User = mongoose.model("User", new mongoose.Schema({
  username: String,
  password: String,
  email: String,
  primaryFingerprint: String,
  primaryDeviceInfo: Object,
}));

const Fingerprint = mongoose.model("Fingerprint", new mongoose.Schema({
  userId: mongoose.Schema.Types.ObjectId,
  timestamp: String,
  fingerprint: String,
  deviceInfo: Object,
}));

// Routes
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "public", "index.html")));
app.get("/login", (req, res) => res.sendFile(path.join(__dirname, "public", "login.html")));
app.get("/verification-sent", (req, res) => res.sendFile(path.join(__dirname, "public", "verification.html")));
app.get("/registered", (req, res) => res.sendFile(path.join(__dirname, "public", "registered.html")));

app.post("/signup", async (req, res) => {
  const { username, password, email, timestamp, fingerprint, deviceInfo } = req.body;

  try {
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return res.status(400).json({ message: "Username already exists" });
    }

    const user = new User({
      username,
      password,
      email,
      primaryFingerprint: fingerprint,
      primaryDeviceInfo: deviceInfo,
    });
    await user.save();

    const fp = new Fingerprint({
      userId: user._id,
      timestamp,
      fingerprint,
      deviceInfo,
    });
    await fp.save();

    res.json({ message: "✅ Sign-up successful" });
  } catch (err) {
    console.error("❌ Error in sign-up:", err);
    res.status(500).json({ message: "Server error during sign-up" });
  }
});

app.post("/login", async (req, res) => {
  const { username, password, timestamp, fingerprint, deviceInfo } = req.body;

  try {
    console.log("🔐 Login attempt:", { username, fingerprint });

    const user = await User.findOne({ username, password });
    if (!user) {
      console.log("❌ No user found for", username);
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const isPrimary = user.primaryFingerprint === fingerprint;
    console.log("🔎 Fingerprint match:", isPrimary);

    const fp = new Fingerprint({
      userId: user._id,
      timestamp,
      fingerprint,
      deviceInfo,
    });
    await fp.save();

    if (isPrimary) {
      return res.json({ message: "✅ Logged in from primary device" });
    } else {
      const code = Math.floor(100000 + Math.random() * 900000).toString();
      console.log("📧 Sending code to:", user.email, "code:", code);
      try {
        await sendVerificationEmail(user.email, code);
        verificationCodes.set(username, code);
        setTimeout(() => verificationCodes.delete(username), 5 * 60 * 1000); // 5 minutes expiry
      } catch (emailErr) {
        console.error("❌ Email send error:", emailErr);
        return res.status(500).json({ message: "Email sending failed" });
      }
      return res.json({ message: "Verification code sent. Please check your email." });
    }
  } catch (err) {
    console.error("❌ Error in login:", err);
    res.status(500).json({ message: "Login failed" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://0.0.0.0:${PORT}`);
});

app.post("/verify", (req, res) => {
  const { username, code } = req.body;
  const savedCode = verificationCodes.get(username);

  if (savedCode && savedCode === code) {
    verificationCodes.delete(username);
    window.location.href = "http://127.0.0.1:5000/";
    return res.json({ success: true, message: "✅ Verification successful!" });
  } else {
    return res.json({ success: false, message: "❌ Invalid or expired code." });
  }
});