# Roblox Screenshare Tool
This is a tool that allows for streaming of video and audio to Roblox. It is meant to be used in require format for things like serversided code executors, but also functions standalone.

## How to use
1. Download VB Audio Cable
2. Run "server.py"
3. Use a service like Cloudflared (cloudflare tunnel) to proxy your local connection through a domain (OR PORT FORWARD, UNSAFE)
 3a. The proxy.py file is supplied to be ran on external devices
4. Run the script below (If the module don't exist/work, you may reupload the rbxm):

```lua
require(140187158857796)(owner.Name, "http://example.com") -- Live Screenshare
```
