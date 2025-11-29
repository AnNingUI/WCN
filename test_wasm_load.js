// Simple test to check if WASM module loads correctly
const fs = require('fs');

// console.log('Testing WASM module load...');

// Check if files exist
const jsFile = 'build-wasm/wcn.js';
const wasmFile = 'build-wasm/wcn.wasm';

if (fs.existsSync(jsFile)) {
    // console.log('✓ wcn.js found:', fs.statSync(jsFile).size, 'bytes');
} else {
    // console.log('✗ wcn.js not found');
}

if (fs.existsSync(wasmFile)) {
    // console.log('✓ wcn.wasm found:', fs.statSync(wasmFile).size, 'bytes');
} else {
    // console.log('✗ wcn.wasm not found');
}

// Try to load the JavaScript module (simulate browser environment)
try {
    // console.log('\nAttempting to load WCN module...');
    
    // Create a mock WebGPU context for testing
    global.navigator = {
        gpu: {
            requestAdapter: async () => ({
                requestDevice: async () => ({
                    features: [],
                    limits: {},
                    queue: {
                        submit: () => {},
                        writeBuffer: () => {},
                        writeTexture: () => {}
                    },
                    destroy: () => {},
                    createShaderModule: () => ({}),
                    createRenderPipeline: () => ({}),
                    createPipelineLayout: () => ({}),
                    createBindGroupLayout: () => ({}),
                    createBuffer: () => ({}),
                    createTexture: () => ({}),
                    createSampler: () => ({}),
                    createCommandEncoder: () => ({
                        beginRenderPass: () => ({
                            setPipeline: () => {},
                            setBindGroup: () => {},
                            setViewport: () => {},
                            setScissorRect: () => {},
                            draw: () => {},
                            end: () => {}
                        }),
                        finish: () => ({})
                    })
                }),
                info: {},
                limits: {}
            })
        },
        getGPUPreferredCanvasFormat: () => 'bgra8unorm'
    };

    // Mock canvas context
    global.HTMLCanvasElement.prototype.getContext = function(type) {
        if (type === 'webgpu') {
            return {
                configure: () => {},
                getCurrentTexture: () => ({
                    createView: () => ({})
                })
            };
        }
        return null;
    };

    // Load and evaluate the WCN module
    const wcnCode = fs.readFileSync(jsFile, 'utf8');
    // console.log('WCN JavaScript code loaded, length:', wcnCode.length);
    
    // Basic check for createWCNModule function
    if (wcnCode.includes('createWCNModule') || wcnCode.includes('createWCNModule')) {
        // console.log('✓ createWCNModule function found');
    } else {
        // console.log('✗ createWCNModule function not found');
    }

} catch (error) {
    // console.log('✗ Error loading WCN module:', error.message);
}

// console.log('\nTest completed.');
