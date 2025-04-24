const nodemailer = require("nodemailer");

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "authenticatordevicefp@gmail.com", // ✅ Your Gmail
    pass: "umflnutcchdztobh",               // ✅ App password from Google
  },
});

const sendVerificationEmail = async (email, code) => {
  const mailOptions = {
    from: "authenticatordevicefp@gmail.com",
    to: email,
    subject: "🔐 Login Verification Code",
    html: `<h2>Your login code: <span style="color:blue;">${code}</span></h2>`,
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log("✅ Verification email sent to:", email);
  } catch (err) {
    console.error("❌ Failed to send verification email:", err);
    throw err;
  }
};

module.exports = sendVerificationEmail;