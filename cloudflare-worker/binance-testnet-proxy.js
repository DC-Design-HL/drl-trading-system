/**
 * Cloudflare Worker - Binance Testnet Proxy
 * Bypasses geo-restrictions by proxying requests through Cloudflare's edge network
 *
 * Deploy this to workers.cloudflare.com to get a proxy URL
 */

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  // Only allow specific origins (your HuggingFace Space)
  const allowedOrigins = [
    'https://chen4700-drl-trading-bot-dev.hf.space',
    'https://chen4700-drl-trading-bot.hf.space',
    'http://localhost:8501', // For local testing
    'http://127.0.0.1:8501'
  ]

  const origin = request.headers.get('Origin')
  const corsHeaders = {
    'Access-Control-Allow-Origin': origin && allowedOrigins.includes(origin) ? origin : allowedOrigins[0],
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-MBX-APIKEY, Authorization',
    'Access-Control-Max-Age': '86400',
  }

  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      headers: corsHeaders
    })
  }

  try {
    const url = new URL(request.url)

    // Extract the target path from the request
    // Format: https://your-worker.workers.dev/api/v3/time
    // Maps to: https://testnet.binance.vision/api/v3/time

    const path = url.pathname + url.search
    const targetUrl = `https://testnet.binance.vision${path}`

    console.log(`Proxying: ${request.method} ${targetUrl}`)

    // Build headers for Binance API
    const headers = new Headers()

    // Copy important headers from original request
    const headersToForward = [
      'X-MBX-APIKEY',
      'Content-Type',
      'User-Agent'
    ]

    for (const header of headersToForward) {
      const value = request.headers.get(header)
      if (value) {
        headers.set(header, value)
      }
    }

    // Make request to Binance testnet
    const response = await fetch(targetUrl, {
      method: request.method,
      headers: headers,
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined
    })

    // Clone response and add CORS headers
    const newResponse = new Response(response.body, response)

    // Add CORS headers to response
    Object.keys(corsHeaders).forEach(key => {
      newResponse.headers.set(key, corsHeaders[key])
    })

    return newResponse

  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Proxy error',
      message: error.message,
      stack: error.stack
    }), {
      status: 500,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    })
  }
}
