import {
    CopilotRuntime,
    copilotRuntimeNextJSAppRouterEndpoint,
    OpenAIAdapter,
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';


const serviceAdapter = new OpenAIAdapter()
//   const runtime = new CopilotRuntime({
//     agents: {
//       // @ts-ignore
//       llamaIndexAgent : llamaIndexAgent 
//     },
//   });
const runtime = new CopilotRuntime({
    remoteEndpoints : [
        {
            url : process.env.NEXT_PUBLIC_SHOPPING_AGENT_URL || "http://localhost:8000/copilotkit",
        }
    ]
})
export const POST = async (req: NextRequest) => {
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
        runtime,
        serviceAdapter,
        endpoint: '/api/copilotkit',
    });

    return handleRequest(req);
};