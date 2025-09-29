"use client"

import { useEffect, useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { Canvas } from "@/components/canvas"
import { WishlistView } from "@/components/wishlist-view"
import { ReportView } from "@/components/report-view"
import { useCoAgent, useCoAgentStateRender, useCopilotAction, useCopilotChat, useCopilotContext, useCopilotMessagesContext } from "@copilotkit/react-core"
import DialogBox from "./tool-response"
import { useCopilotChatSuggestions } from "@copilotkit/react-ui"
import { ToolLog, ToolLogs } from "./tool-logs"
import { TextMessage, ActionExecutionMessage, ResultMessage, AgentStateMessage, Role, Message } from "@copilotkit/runtime-client-gql"

const mockProducts = [
  {
    id: "1",
    name: 'Apple MacBook Pro 14"',
    price: "$1,999",
    image: "/placeholder.svg?height=200&width=200&text=MacBook+Pro",
    pros: [
      "M3 Pro chip performance",
      "Excellent Liquid Retina display",
      "18-hour battery life",
      "Premium build quality",
    ],
    cons: ["Expensive price point", "Limited port selection", "No touchscreen"],
    source: "Apple Store",
    rating: 4.8,
    reviews: 1247,
    specs: {
      processor: "Apple M3 Pro",
      ram: "18GB Unified Memory",
      storage: "512GB SSD",
      display: '14.2" Liquid Retina XDR',
      battery: "Up to 18 hours",
      weight: "3.5 lbs",
      ports: "3x Thunderbolt 4, HDMI, SD card",
      os: "macOS Sonoma",
    },
  },
  {
    id: "2",
    name: "Dell XPS 13 Plus",
    price: "$1,299",
    image: "/placeholder.svg?height=200&width=200&text=Dell+XPS+13",
    pros: ["Sleek modern design", "Good performance for price", "4K display option", "Compact form factor"],
    cons: ["Average 10-hour battery", "Unique keyboard layout", "Limited upgradeability"],
    source: "Dell.com",
    rating: 4.5,
    reviews: 892,
    specs: {
      processor: "Intel Core i7-1360P",
      ram: "16GB LPDDR5",
      storage: "512GB SSD",
      display: '13.4" 4K+ Touch',
      battery: "Up to 10 hours",
      weight: "2.73 lbs",
      ports: "2x Thunderbolt 4",
      os: "Windows 11",
    },
  },
  {
    id: "3",
    name: "Lenovo ThinkPad X1 Carbon",
    price: "$1,599",
    image: "/placeholder.svg?height=200&width=200&text=ThinkPad+X1",
    pros: ["Military-grade durability", "Excellent keyboard", "Business security features", "15-hour battery life"],
    cons: ["Higher price than competitors", "Average display quality", "Conservative design"],
    source: "Lenovo.com",
    rating: 4.6,
    reviews: 634,
    specs: {
      processor: "Intel Core i7-1365U",
      ram: "16GB LPDDR5",
      storage: "1TB SSD",
      display: '14" WUXGA IPS',
      battery: "Up to 15 hours",
      weight: "2.48 lbs",
      ports: "2x Thunderbolt 4, 2x USB-A, HDMI",
      os: "Windows 11 Pro",
    },
  },
]

const mockSuggestions = [
  "Find laptops under $1500",
  "Compare gaming headphones",
  "Show wireless earbuds with ANC",
  "Best smartphones for photography",
  "Budget 4K monitors under $400",
]

interface ChatSession {
  id: number
  name: string
  timestamp: Date
  messageCount: number
  lastActivity: Date
}

const initialConvo = [{
  conversationId: 1,
  messages: [],
  chatName: "Conversation 1",
  state: {
    products: [],
    favorites: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
    wishlist: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
    buffer_products: [],
    logs: [] as ToolLog[],
    report: null,
    show_results: false,
    canvas_logs: {
      title: "",
      subtitle: ""
    }
  }
}]

export function ShoppingAssistant() {
  const [query, setQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const { setThreadId } = useCopilotContext()
  const [isChatCreated, setIsChatCreated] = useState(false)
  const [conversationHistory, setConversationHistory] = useState<any>(typeof window !== 'undefined' && window.localStorage.getItem("conversationHistory") ? (() => {
    let stored = JSON.parse(window.localStorage.getItem("conversationHistory") || "[]");
    console.log(stored, "storedstoredstored");

    let fullMessages: any[] = [];
    for (const conversation of stored) {
      let finalMessages = [];
      for (const message of conversation?.messages) {
        if (message?.type === "TextMessage") {
          finalMessages.push(new TextMessage({
            role: message?.role === "user" ? Role.User : Role.Assistant,
            content: message?.content
          }));
        }
        else if (message?.type === "ActionExecutionMessage") {
          finalMessages.push(new ActionExecutionMessage({
            name: message?.name,
            arguments: message?.arguments
          }));
        }
        else if (message?.type === "ResultMessage") {
          finalMessages.push(new ResultMessage({
            actionExecutionId: message?.actionExecutionId,
            actionName: message?.actionName,
            result: message?.result
          }));
        }
        else if (message?.type === "AgentStateMessage") {
          finalMessages.push(new AgentStateMessage({
            agentName: message?.agentName,
            state: message?.state
          }));
        }
      }
      fullMessages.push(finalMessages);
    }
    stored.forEach((conversation: any, index: number) => {
      conversation.messages = fullMessages[index];
    });
    console.log("storedprocessed", stored);

    return stored;
  })() : initialConvo)
  const { visibleMessages, isLoading, reset } = useCopilotChat()
  const [currentChatId, setCurrentChatId] = useState<number>(conversationHistory.length > 0 ? conversationHistory[0]?.conversationId : 1)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([{
    id: 1,
    name: "New Chat",
    timestamp: new Date(),
    messageCount: 0,
    lastActivity: new Date()
  }])
  // const [products, setProducts] = useState<any>(mockProducts)
  const { state, setState, start, run } = useCoAgent({
    name: "shopping_agent_crewai",
    initialState: conversationHistory.length > 0 ? { ...conversationHistory[0]?.state, show_results: (conversationHistory[0]?.state?.products?.length > 0 ? true : false) } : {
      products: [],
      favorites: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
      wishlist: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
      buffer_products: [],
      logs: [] as ToolLog[],
      report: null,
      show_results: false,
      canvas_logs: {
        title: "",
        subtitle: ""
      }
    }
  })
  const { messages, setMessages } = useCopilotMessagesContext();
  // useEffect(() => {
  //   // debugger
  //   console.log(conversationHistory[0], "conversationHistory");
  //   // console.log(JSON.parse(window.localStorage.getItem("conversationHistory") || "[]")[0]?.messages, "conversationHistory");
  //   // setState(conversationHistory[0]?.state)
  //   // setCurrentChatId(conversationHistory[0]?.conversationId)
  //   // setMessages([new TextMessage({
  //   //   role: Role.User,
  //   //   content: "Hello, how are you?",
  //   // })])
  //   if ((conversationHistory[0]?.messages?.length > 0 && conversationHistory[conversationHistory.length - 1]?.messages?.length > 0)) {
  //     console.log("Setting message here");

  //     setMessages(conversationHistory[0]?.messages)
  //   }


  //   // setMessages(JSON.parse(window.localStorage.getItem("conversationHistory") || "[]")[0]?.messages)
  // }, [conversationHistory])


  useEffect(() => {
    debugger
    let index = conversationHistory.findIndex((conversation: any) => conversation.conversationId === currentChatId)
    if (index != -1) {
      let modifiedConversation = conversationHistory
      modifiedConversation[index].messages = messages
      modifiedConversation[index].state = state
      setConversationHistory(modifiedConversation)
    }

  }, [messages, currentChatId])


  useEffect(() => {
    console.log(conversationHistory, "conversationHistory");
    if ((conversationHistory[0]?.messages?.length > 0 && conversationHistory[conversationHistory.length - 1]?.messages?.length > 0)) {
      console.log("Setting message here");

      setMessages(conversationHistory[0]?.messages)
    }
    if (typeof window === 'undefined') return;

    const handleBeforeUnload = () => {
      if(window.localStorage.getItem("conversationHistory")=="[]"){
        window.localStorage.setItem("conversationHistory", JSON.stringify(initialConvo))
        return
      } 
      window.localStorage.setItem("conversationHistory", JSON.stringify(conversationHistory));
      // window.localStorage.setItem("wishlist", JSON.stringify(state?.favorites));
    };

    // Runs when user closes tab or refreshes
    window.addEventListener("beforeunload", handleBeforeUnload);
    if (isChatCreated) {
      window.location.reload()
    }
    // Cleanup
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [conversationHistory]);




  // const [wishlist, setWishlist] = useState(window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [])

  const [currentView, setCurrentView] = useState<"products" | "wishlist" | "report">("products")

  // Chat management functions
  const generateChatName = (firstMessage: string) => {
    const words = firstMessage.split(' ').slice(0, 4).join(' ')
    return words.length > 30 ? words.substring(0, 30) + '...' : words
  }

  const handleCreateNewChat = (from = "not-del") => {
    debugger
    const newChatId = from != "del" ? Math.max(...conversationHistory.map((c: any) => c.conversationId)) + 1 : 1
    const newChat = {
      conversationId: newChatId,
      chatName: "Conversation " + newChatId,
      messages: [],
      state: {
        products: [],
        favorites: [] as string[],
        wishlist: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
        buffer_products: [],
        logs: [] as ToolLog[],
        report: null,
        show_results: false,
        canvas_logs: {
          title: "",
          subtitle: ""
        }
      }
    }
    setIsChatCreated(true)
    setConversationHistory([newChat, ...conversationHistory])
    setCurrentChatId(newChatId)

    // Reset current state
    setState({
      products: [],
      favorites: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
      wishlist: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [],
      buffer_products: [],
      logs: [] as ToolLog[],
      report: null,
      show_results: false,
      buffer_messages: [],
      canvas_logs: {
        title: "",
        subtitle: ""
      }
    })

    // Clear messages and query
    setMessages([])
    setThreadId(newChatId.toString())
    setQuery("")
    setCurrentView("products")
  }

  const handleSwitchChat = (chatId: number) => {
    debugger
    const conversationIndex = conversationHistory.findIndex((conv: any) => conv.conversationId === chatId)
    if (conversationIndex !== -1) {
      const conversation = conversationHistory[conversationIndex]
      setCurrentChatId(chatId)
      setMessages(conversation.messages || [])
      // setFavorites(typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [])
      setState({ ...conversation.state, show_results: (conversation.state.products.length > 0 ? true : false), favorites: typeof window !== 'undefined' && window.localStorage.getItem("wishlist") ? JSON.parse(window.localStorage.getItem("wishlist") || "[]") : [] })
      setCurrentView("products")
    }
  }

  const handleRenameChat = (chatId: number, newName: string) => {
    setConversationHistory((prev: any) => prev.map((chat: any) =>
      chat.conversationId === chatId ? { ...chat, chatName: newName } : chat
    ))
  }

  const handleDeleteChat = (chatId: number) => {
    debugger
    // if (conversationHistory.length <= 1) return // Don't delete the last chat

    // setChatSessions(prev => prev.filter(chat => chat.id !== chatId))
    setConversationHistory((prev: any) => prev.filter((conv: any) => conv.conversationId !== chatId))

    // If we're deleting the current chat, switch to another one
    if (currentChatId === chatId) {
      const remainingChats = conversationHistory.filter((chat: any) => chat.conversationId !== chatId)
      if (remainingChats.length > 0) {
        handleSwitchChat(remainingChats[0].conversationId)
      }
      else {
        if (typeof window !== 'undefined') {
          window.localStorage.setItem("conversationHistory", JSON.stringify([]))
          window.location.reload()
        }
      }
    }
  }

  const toggleWishlist = (productId: string) => {
    debugger
    console.log('state?.favorites', state);

    if (state?.favorites.map((id: any) => id.id).includes(productId)) {
      setState({
        ...state,
        favorites: state?.favorites?.filter((product: any) => product.id !== productId),
        // wishlist: state?.wishlist?.filter((product: any) => product.id !== productId)
      })
      if (typeof window !== 'undefined') {
        window.localStorage.setItem("wishlist", JSON.stringify(state?.favorites?.filter((product: any) => product.id !== productId)))
      }
      setConversationHistory((prev: any) => prev.map((conversation: any) => conversation.conversationId === currentChatId ? { ...conversation, state: { ...conversation.state, favorites: state?.favorites?.filter((product: any) => product.id !== productId) } } : conversation))
    }
    else {
      setState({
        ...state,
        favorites: [...state?.favorites, ...state?.products?.filter((product: any) => product.id === productId)],
        // wishlist: [...state?.wishlist, ...state?.products?.filter((product: any) => product.id === productId)]
      })
      if (typeof window !== 'undefined') {
        window.localStorage.setItem("wishlist", JSON.stringify([...state?.favorites, ...state?.products?.filter((product: any) => product.id === productId)]))
      }
      setConversationHistory((prev: any) => prev.map((conversation: any) => conversation.conversationId === currentChatId ? { ...conversation, state: { ...conversation.state, favorites: [...state?.favorites, ...state?.products?.filter((product: any) => product.id === productId)] } } : conversation))
    }
  }

  const deleteProduct = (productId: string) => {
    const productToDelete = state?.products?.find((p: any) => p.id === productId)
    if (!productToDelete) return

    if (state?.buffer_products?.length > 0) {
      let a = state?.buffer_products.pop()
      setState({
        ...state,
        products: [...state?.products?.filter((p: any) => p.id !== productId), ...(a ? [a] : [])]
      })
    }
    else {
      setState({
        ...state,
        products: state?.products?.filter((p: any) => p.id !== productId)
      })
    }
  }

  // useEffect(() => {
  //   setWishlist(state?.products?.filter((product: any) => state?.favorites?.includes(product.id)))
  //   window.localStorage.setItem("wishlist", JSON.stringify(state?.products?.filter((product: any) => state?.favorites?.includes(product.id))))
  // }, [state?.products, state?.favorites])

  const goToReport = () => {
    setCurrentView("report")
  }

  const goToWishlist = () => {
    setCurrentView("wishlist")
  }

  const exitToProducts = () => {
    setCurrentView("products")
  }

  const handleSearch = async (searchQuery: string) => {
    setIsSearching(true)
    setQuery(searchQuery)

    // Set agent decisions based on search

    // Simulate API call
    setTimeout(() => {
      if (searchQuery.toLowerCase().includes("laptop") || searchQuery.toLowerCase().includes("computer")) {
        // setProducts(mockProducts)
      }
      setIsSearching(false)
    }, 2000)
  }





  const wishlistProducts = state?.products?.filter((product: any) => state?.favorites?.includes(product.id))

  useCoAgentStateRender({
    name: "shopping_agent_crewai",
    render: (state1: any) => {
      // useEffect(() => {
      // console.log(state1, "state1")
      // }, [state1])

      return <ToolLogs logs={state1?.state?.logs || []} />
    }
  })


  useCopilotAction({
    name: "edit_product_canvas",
    description: "Ability to edit the products like moving it to wishlist or removing it from playlist or removing it from the product canvas",
    parameters: [
      {
        name: "remove_from_canvas",
        type: "object[]",
        description: "When user explicity ask to remove the product from the canvas, you need set values. Otherwise, set values as empty array",
        attributes: [
          {
            name: "product_id",
            type: "string",
            description: "The id of the product to be removed from the canvas"
          }
        ]
      },
      {
        name: "move_to_wishlist",
        type: "object[]",
        description: "When user explicity ask to move the product to wishlist, you need set values. Otherwise, set values as empty array",
        attributes: [
          {
            name: "product_id",
            type: "string",
            description: "The id of the product to be moved to wishlist"
          }
        ]
      },
      {
        name: "remove_from_wishlist",
        type: "object[]",
        description: "When user explicity ask to remove the product from wishlist, you need set values. Otherwise, set values as empty array",
        attributes: [
          {
            name: "product_id",
            type: "string",
            description: "The id of the product to be removed from wishlist"
          }
        ]
      }
    ],
    handler: (args) => {
      console.log(args, "argsargsargsargs")
      debugger
      if (args?.move_to_wishlist?.length > 0) {
        setState({
          ...state,
          favorites: [...state?.favorites, ...state?.products?.filter((product: any) => args?.move_to_wishlist?.map((product: any) => product?.product_id)?.includes(product?.id))]
        })
        window.localStorage.setItem("wishlist", JSON.stringify([...state?.favorites, ...state?.products?.filter((product: any) => args?.move_to_wishlist?.map((product: any) => product?.product_id)?.includes(product?.id))]))
        return "Product Moved to wishlist successfully"
      }
      if (args?.remove_from_wishlist?.length > 0) {
        let itemsToRemove = args?.remove_from_wishlist?.map((product: any) => product?.product_id)
        setState({
          ...state,
          favorites: state?.favorites?.filter((product: any) => !itemsToRemove?.includes(product?.id))
        })
        window.localStorage.setItem("wishlist", JSON.stringify(state?.favorites?.filter((product: any) => !itemsToRemove?.includes(product?.id))))
        return "Product Removed from wishlist successfully"
      }
      if (args?.remove_from_canvas?.length > 0) {
        debugger
        let itemsToRemove = args?.remove_from_canvas?.map((product: any) => product?.product_id)
        if (state?.buffer_products?.length > 0) {
          let a = state?.buffer_products.pop()
          setState({
            ...state,
            products: [...state?.products?.filter((p: any) => !itemsToRemove?.includes(p?.id)), ...(a ? [a] : [])]
          })
        }
        else {
          setState({
            ...state,
            products: state?.products?.filter((p: any) => !itemsToRemove?.includes(p?.id))
          })
        }
        return "Product Removed from canvas successfully"
      }
    }
  })

  useEffect(() => {
    console.log(state, "statestatestatestate")
  }, [state])

  useCopilotAction({
    name: "list_products",
    description: "A list of products that are scraped from web",
    renderAndWaitForResponse: ({ status, respond, args }) => {
      // console.log(args, "argsargsargsargs")

      return <DialogBox isDisabled={respond == undefined} contentList={args?.products?.map((product: any) => ({ title: product.title, url: product.product_url }))}
        onAccept={() => {
          debugger
          if (respond) {
            respond("Accepted")
            setState({
              ...state,
              products: args?.products,
              buffer_products: args?.buffer_products.slice(5, args?.buffer_products.length),
              logs: []
            })
            let conversations = conversationHistory
            conversations.forEach((conversation: any) => {
              if (conversation.conversationId == currentChatId) {
                conversation.chatName = args?.chat_name
              }
            })
            setConversationHistory(conversations)
            setCurrentChatId(currentChatId)
            console.log(currentChatId, "currentChatId");
            // setConversationHistory((prev: any) => prev.map((conversation: any) => conversation.conversationId === currentChatId ? { ...conversation, chatName: args?.chat_name } : conversation))
            // setProducts(args?.products)
          }
        }}
        onReject={() => {
          if (respond) {
            respond("Rejected")
            setState({
              ...state,
              logs: []
            })
          }
        }}
        onNeedInfo={() => {
          debugger
          if (respond) {
            respond("Show more products")
            setState({
              ...state,
              products: args?.buffer_products?.slice(0, 10),
              buffer_products: args?.buffer_products.slice(10, args?.buffer_products.length),
              logs: []
            })

            let conversations = conversationHistory
            conversations.forEach((conversation: any) => {
              if (conversation.conversationId === currentChatId) {
                conversation.chatName = args?.chat_name
              }
            })
            console.log(currentChatId, "currentChatId");

            setConversationHistory(conversations)
            setCurrentChatId(currentChatId)
            // setConversationHistory((prev: any) => prev.map((conversation: any) => conversation.conversationId === currentChatId ? { ...conversation, chatName: args?.chat_name } : conversation))
            // setProducts(args?.buffer_products?.slice(0, 10))
          }
        }} />
    }

  })

  useCopilotChatSuggestions({
    available: "enabled",
    instructions: "You need to provide suggestions for the user to buy products like laptops, phones, headphones, etc. Example suggestions: Find laptops under $1500, Find good smartphones for photography, Get me some wireless earbuds with ANC, Best smartphones under $500, Budget 4K monitors under $400",
  })

  useEffect(() => {
    console.log(visibleMessages, "visible");

  })

  useEffect(() => {
    console.log(visibleMessages.filter((message: any) => message?.role === "user"))
    // if (state?.show_results) {
    // @ts-ignore
    setQuery(visibleMessages.filter((message: any) => message?.role === "user")[visibleMessages.filter((message: any) => message?.role === "user").length - 1]?.content)
    // }
    // visibleMessages.filter((message : any) => message.type)
  }, [isLoading])


  return (
    <div className="flex h-screen bg-[#FAFCFA] overflow-hidden">
      <Sidebar
        setQuery={setQuery}
        clearState={setState}
        onSearch={handleSearch}
        suggestions={mockSuggestions}
        currentQuery={query}
        isLoading={isLoading}
        isSearching={isSearching}
        currentView={currentView}
        wishlistCount={state?.favorites?.length}
        goToProducts={exitToProducts}
        currentChatId={currentChatId}
        chatSessions={conversationHistory}
        onSwitchChat={handleSwitchChat}
        onCreateNewChat={handleCreateNewChat}
        onRenameChat={handleRenameChat}
        onDeleteChat={handleDeleteChat}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {currentView === "report" ? (
          <ReportView isLoading={isLoading} products={state?.products} onExit={exitToProducts} searchQuery={query} report={state?.report} />
        ) : currentView === "wishlist" ? (
          <WishlistView
            clearAllWishlist={() => {
              debugger
              setState({
                ...state,
                favorites: []
              })
              setConversationHistory((prev: any) => prev.map((conversation: any) => conversation.conversationId === currentChatId ? { ...conversation, state: { ...conversation.state, favorites: [] } } : conversation))
              if (typeof window !== 'undefined') {
                window.localStorage.setItem("wishlist", JSON.stringify([]))
              }
            }}
            products={state?.favorites}
            onExit={exitToProducts}
            onToggleWishlist={toggleWishlist}
            onDeleteProduct={deleteProduct}
          />
        ) : (
          <Canvas
            canvasLogs={state?.canvas_logs}
            start={run}
            show_results={state?.show_results}
            report={state?.report}
            products={state?.products}
            isLoading={isLoading && !state?.show_results}
            query={query}
            wishlistLength={state?.favorites?.length}
            wishlist={state?.favorites}
            onToggleWishlist={toggleWishlist}
            onDeleteProduct={deleteProduct}
            onGoToWishlist={goToWishlist}
            onGoToReport={goToReport}
          />
        )}
      </div>
    </div>
  )
}
