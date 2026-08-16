// tsx asks os.userInfo() only to name its temporary directory. Some restricted
// Windows processes cannot resolve the account through libuv, so provide the
// same stable identifier through the Unix-style hook that tsx checks first.
if (process.platform === "win32" && typeof process.geteuid !== "function") {
  process.geteuid = () => process.env.USERNAME || "windows";
}
